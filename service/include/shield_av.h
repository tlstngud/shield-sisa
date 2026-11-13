/*
 * shield_av.h -- SHIELD-AV live service core (C API), build/service/libshield_av.so
 *
 * One session per call participant accepts audio PCM and decoded video frames at the same time and returns, every 2 s,
 * one combined result with the audio result (SHIELD-A, via the native audio engine's shield_stream, low-pass ON) and the
 * video result (SHIELD-V, native preprocessing + CUDA or CPU model), each with its own status, plus a combined alert
 * from a stated, configurable policy.  It is not a learned fusion: one modality never fills in the other's score.
 *
 * TIMELINE (per session)
 *   Session time 0 is the first audio sample; audio sample i is at i/16000 s.  Hop h (h = 0, 1, 2, ...) ends at session
 *   time 2(h+1) and covers the 10 s window [2h-8, 2h+2).  Audio hop h is complete once 32000(h+1) samples were pushed
 *   (shield_stream semantics: collecting while fewer than 10 s were pushed, i.e. h < 4).
 *   Video frames are pushed with timestamps in seconds on the same session clock, strictly increasing.  The video
 *   window of hop h samples the first frame at/after each target 2h-8 + i/2 (i = 0..19), exactly the reference
 *   sample_frames rule (1e-6 slack; a frame more than 0.25 s late makes the target missing and every window that
 *   contains it unavailable).  Internally the session runs the preprocessing library's live participant helper with
 *   start = (first window start) - (first frame time), so the 20 crops of a hop are those of the reference
 *   video_preprocess.prepare_video on the same frames.
 *   Reference mode (session config video_start finite, for parity tests with files): the targets are
 *   video_start + k/2 seconds after the FIRST PUSHED FRAME (exactly sample_frames(window_start = video_start + 2w)
 *   for participant window w), and participant window w is reported as hop video_first_hop + w.
 *
 * WHEN A HOP IS EMITTED
 *   Results are emitted strictly in hop order, once both modality parts of the hop are final:
 *   - audio part: final when the audio engine returned the hop (immediately for collecting / insufficient speech);
 *   - video part: final when the hop's video window resolved (gap -> unavailable; complete -> scored) and, if scored,
 *     the model returned.  If the audio clock reaches hop_end + video_wait_s and the window has not resolved (video
 *     stalled or lagging), the video part becomes unavailable (reason SHAV_R_LAG).  Before the first video frame,
 *     and for hops whose window starts before the first video target, the video part is collecting.
 *   - AUDIO LAG RULE (the mirror of the video one): if the video clock (timestamp of the newest pushed frame, session
 *     seconds) reaches hop_end + audio_wait_s and the hop's audio INPUT has not arrived (microphone muted, dropped
 *     or stalled), the audio part becomes final without a score: unavailable / SHAV_R_LAG (collecting /
 *     SHAV_R_NO_INPUT if no audio sample was ever pushed; collecting / SHAV_R_WARMUP for hops 0..3), logit/score
 *     NaN.  The hop, with its own video result, is then emitted in order: a stalled microphone cannot hold back
 *     video results.  Only hops whose video window has resolved are forced, and never a hop whose audio input is
 *     complete and only being computed (audio compute latency is not lag).  audio_wait_s defaults to video_wait_s
 *     (session config, NaN = mirror; +inf turns the rule off).
 *     Consequences: audio that arrives later for a forced hop still runs through the audio stream (its ring and
 *     low-pass need every sample) but its result is discarded, and the audio engine's own k-of-n history still
 *     counts it.  Audio hop numbering is by sample count (sample i is at i/16000 s): after a gap in the audio
 *     input, push silence for the missing span (or reset) to keep the audio on the session clock, otherwise every
 *     later audio hop stays behind the video clock and is reported lag.  An audio producer that pushes in chunks
 *     longer than audio_wait_s while video runs ahead must raise audio_wait_s.
 *   - a disabled modality (session config) or a missing model (engine) is unavailable (reason SHAV_R_DISABLED).
 *   - audio-disabled sessions: the hop clock is the video (hops are emitted as their video windows resolve).
 *   shav_session_finish() declares the end of input: unresolved video windows become unavailable (SHAV_R_ENDED), and
 *   so does the audio part of every hop whose video window resolved but whose audio input is incomplete (audio
 *   SHAV_R_ENDED), all jobs are awaited and every hop that either modality reached is emitted.
 *
 * STATUSES / NaN
 *   SHAV_COLLECTING, SHAV_VALID, SHAV_UNAVAILABLE for both modalities, plus SHAV_INSUFFICIENT_SPEECH for audio (the
 *   audio engine's VAD gate: fewer than 3 s of speech in the window).  logit and score are NaN unless VALID
 *   (serialise as null, never 0); decision and alert are -1 unless VALID.  A valid video result carries
 *   face_detection_anchors (0..5); zero anchors keeps the reference whole-frame fallback.
 *
 * DECISIONS AND ALERTS
 *   score = sigmoid(logit) in double.  Video decision: score >= video threshold (read from "<video weights>.json").
 *   The model libraries refuse a sidecar whose "output_sha256" is not the weights' SHA-256 or whose threshold is
 *   missing / non-finite / outside (0, 1), and every certificate line binds the threshold it was issued for: an
 *   edited threshold loads only the exact tiers (CUDA "fp32", CPU "exact"); in the default CUDA mode the engine
 *   then falls back to fp32 and load_log says why.  The threshold in use is shav_engine_info.video_threshold.
 *   Audio decision: score >= session audio_threshold.  Each modality's alert = at least k of its last n valid
 *   decisions (audio: the audio engine's own k-of-n history; video: the same rule in the session).
 *   Combined alert (session config policy):
 *     SHAV_POLICY_EITHER (default): 1 if any valid modality alerts; 0 if at least one modality is valid and none
 *                                   alerts; -1 if neither is valid.
 *     SHAV_POLICY_BOTH:   1 if both are valid and both alert; 0 if some valid modality does not alert; else -1.
 *     SHAV_POLICY_AUDIO / SHAV_POLICY_VIDEO: that modality's alert (-1 unless valid).
 *
 * SCHEDULING
 *   Audio: shield_stream_submit_hop on the audio engine's worker pool (audio_workers, 1..8), driven by the same
 *   number of service driver threads.  Video: CUDA backend = the CUDA library's asynchronous job API (its scheduler
 *   coalesces queued jobs of all sessions FIFO into GPU batches, bounded batching delay cuda_max_delay_us), one job per
 *   window = the crops that are not in the session's exact token cache + 1 window whose other frames read device token
 *   slots; CPU backend = cpu_video_workers threads calling the CPU library.  Backpressure: at most one job in flight per
 *   session per modality.  Audio hops queue behind it (never dropped: the audio ring needs every hop); at most
 *   video_max_pending ready video windows wait, a further one is dropped (unavailable, SHAV_R_OVERLOAD).  Every engine
 *   used is row-independent and session state is private, so a session's results never depend on who else is served.
 *   A failed video job poisons its token-cache generation: pending windows of that generation that read cached
 *   tokens are reported failed, and the next window starts with an empty cache.
 *   Crop staging buffers (20 crops, 3.46 MB; pinned host memory on CUDA) come from an engine pool; each video session
 *   adds one at create, more are allocated on demand for pending windows and kept until the engine is freed.
 *
 * VIDEO PREPROCESSING runs inside shav_session_push_video_* on the calling thread: per sampled frame (2 per second)
 * letterbox + gray + Haar once (cached per frame); when a window completes, its crop rectangles are computed and only
 * the crops missing from the exact token cache are extracted (lazy native canvas), then the frames no later window
 * needs are released (at most 20 cached sampled frames; 1080p input in the default AUTO storage keeps the 1280x720
 * RGB `small` frame, 2.76 MB each).  Non-sampled frames cost a timestamp comparison.  Call it from the decoder threads;
 * one call at a time per session.
 * EXACT TOKEN CACHE: up to 24 entries per session keyed by (sampled frame id, crop rectangle); a window reuses the
 * tokens of every (frame, rect) it shares with earlier windows (CNN features are a pure function of the crop bytes).
 * Measured on all windows of the 840 RTFS val+test videos: 9.8% of frame encodes are reused (7,862 / 79,960); the
 * rectangle usually moves because the anchors are re-interpolated per window.  Results are bitwise identical with and
 * without the cache (docs/PERFORMANCE.md).
 *
 * THREADING
 *   Engine functions are thread-safe.  Different sessions may be used from different threads concurrently.  Per
 *   session, push_audio and push_video may run concurrently with each other (one thread each); other calls on one
 *   session must not overlap.  Results are delivered through the session callback (if set; called in hop order from
 *   a service thread or from inside a push/finish call, must be short and must not call shav_session_* of the same
 *   session) or else queued for shav_session_poll.  Free sessions before the engine.
 *   All functions returning int return 0 on success, -1 on error (shav_last_error(), thread-local).
 */
#ifndef SHIELD_AV_H
#define SHIELD_AV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#define SHAV_API_VERSION 2   /* 2: shav_session_config.audio_wait_s (audio lag rule) */
#define SHAV_SAMPLE_RATE 16000
#define SHAV_HOP_SAMPLES 32000
#define SHAV_HOP_SECONDS 2.0
#define SHAV_WINDOW_SECONDS 10.0

/* statuses */
#define SHAV_COLLECTING 0
#define SHAV_VALID 1
#define SHAV_UNAVAILABLE 2
#define SHAV_INSUFFICIENT_SPEECH 3   /* audio only */

/* reasons (why a part is not valid) */
#define SHAV_R_NONE 0
#define SHAV_R_WARMUP 1       /* collecting: the window starts before the modality's first input / fewer than 10 s */
#define SHAV_R_NO_INPUT 2     /* collecting: no video frame received yet (video), or no audio sample yet while the
                                 video clock passed hop_end + audio_wait_s (audio) */
#define SHAV_R_GAP 3          /* video: a target had no frame within 0.25 s (frame gap; lost/undecodable frames are simply
                                 not pushed) */
#define SHAV_R_LAG 4          /* video: window not resolved when the audio clock reached hop_end + video_wait_s;
                                 audio: audio input not complete when the video clock reached hop_end + audio_wait_s */
#define SHAV_R_DISABLED 5     /* modality disabled in the session or no model in the engine */
#define SHAV_R_FAILED 6       /* model / preprocessing failure, or a non-finite logit */
#define SHAV_R_OVERLOAD 7     /* video window dropped by backpressure */
#define SHAV_R_SILENCE 8      /* audio: insufficient speech (VAD) */
#define SHAV_R_ENDED 9        /* input ended (shav_session_finish) before the window completed (video), or before the
                                 hop's audio input was complete while its video window had resolved (audio) */

/* combined-alert policies */
#define SHAV_POLICY_EITHER 0
#define SHAV_POLICY_BOTH 1
#define SHAV_POLICY_AUDIO 2
#define SHAV_POLICY_VIDEO 3

typedef struct shav_engine shav_engine;
typedef struct shav_session shav_session;

typedef struct shav_modality_result {
    int status, reason;
    float logit;              /* NaN unless VALID */
    double score;             /* sigmoid(logit) in double, NaN unless VALID */
    int decision;             /* score >= threshold: 1/0, -1 unless VALID */
    int alert;                /* k-of-n alert of this modality: 1/0, -1 unless VALID */
    double threshold;
    double window_start, window_end;   /* session seconds ([2h-8, 2h+2) for hop h) */
    double valid_speech_sec;  /* audio VAD speech seconds (NaN for video) */
    int face_detection_anchors;        /* video, VALID: 0..5; else -1 */
    int frames_encoded;       /* video: crops encoded for this window (token-cache misses); -1 if not scored */
    int frames_reused;        /* video: frames whose tokens came from the exact token cache; -1 if not scored */
    double t_input;           /* CLOCK_MONOTONIC s when this part's input was complete (NaN if never) */
    double t_done;            /* CLOCK_MONOTONIC s when this part became final */
    float queue_ms;           /* engine queueing (audio pool / GPU scheduler / CPU worker queue), NaN if none */
    float compute_ms;         /* engine compute (audio job / GPU batch run / CPU call), NaN if none */
    int batch_windows;        /* video GPU: windows in the coalesced GPU batch (0 otherwise) */
} shav_modality_result;

typedef struct shav_hop_result {
    int64_t hop;
    double hop_end;           /* session seconds = 2(hop+1) */
    int alert;                /* combined policy alert: 1/0/-1 */
    int policy;
    shav_modality_result audio, video;
    double t_emit;            /* CLOCK_MONOTONIC s when emitted */
} shav_hop_result;

typedef void (*shav_result_callback)(const shav_hop_result* result, void* user);
/* Test hook: called inside push_video with the 20 crops (180x320x3 uint8 each) of every window that is scored,
 * before inference.  targets: sampled-frame ids; times: seconds after the first pushed frame; rects: 320x180 units. */
typedef struct shav_window_view {
    int64_t hop, window;
    int anchors;
    int64_t targets[20];
    double times[20];
    int rects[20][4];
    const uint8_t* crops;     /* 20 x 172800 bytes */
} shav_window_view;
typedef void (*shav_crop_tap)(const shav_window_view* view, void* user);

typedef struct shav_engine_config {
    const char* audio_weights;     /* exported SHIELD-A weights (SHLDA001); NULL/"" = no audio model */
    const char* audio_mode;        /* NULL = "exact" (others need the audio engine's certificate) */
    int audio_workers;             /* audio engine worker pool, 1..8 (default 4) */
    int audio_cache_level;         /* shield_stream_set_cache_level, -1 = engine default (1) */
    const char* video_weights;     /* SHLDV001 weights (+ .json threshold); NULL/"" = no video model */
    const char* video_backend;     /* "auto" (CUDA if it loads, else CPU; default), "cuda", "cpu", "none" */
    const char* video_mode;        /* NULL = backend default: CUDA "fp16" (certified; falls back to "fp32" if its
                                      certificate is missing or does not cover the sidecar threshold), CPU "exact".
                                      Explicit modes never fall back. */
    int cuda_device;               /* after CUDA_VISIBLE_DEVICES; default 0 */
    int cuda_max_delay_us;         /* GPU scheduler batching delay when idle (default 0) */
    int cuda_max_frames, cuda_max_windows;   /* GPU batch limits (default 640 / 512) */
    int cuda_blocking_sync;        /* 1: set the device's primary CUDA context to blocking synchronisation before the
                                      CUDA library creates it (its worker sleeps instead of spinning while the GPU
                                      runs; process-wide for that device).  0: driver default.  See docs/PERFORMANCE.md */
    int max_sessions;              /* sizes the device token-slot pool: 24 slots per session (default 512) */
    int cpu_video_workers;         /* CPU backend worker threads (default 4) */
    int cpu_video_threads;         /* CPU library team size per call (default 1) */
    const char* cpu_isa;           /* CPU backend dispatch level, NULL = automatic */
    const char* version;           /* service version label (default "shield-av-1") */
} shav_engine_config;

typedef struct shav_session_config {
    const char* participant_id;    /* required, copied */
    int audio_enabled, video_enabled;          /* default 1, 1 */
    double audio_threshold;        /* default 0.5 */
    int audio_alert_k, audio_alert_n;          /* default 1, 1 */
    int audio_lowpass;             /* default 1 (required by the published SHIELD-A weights) */
    int video_alert_k, video_alert_n;          /* default 1, 1 */
    int policy;                    /* default SHAV_POLICY_EITHER */
    double video_start;            /* NaN (default) = session-aligned; finite = reference mode (see TIMELINE) */
    int64_t video_first_hop;       /* reference mode: hop of participant window 0 (default 4) */
    double video_wait_s;           /* default 2.0 */
    int video_cache;               /* exact (frame, rect) token cache, default 1 */
    int video_store;               /* frame storage of the preprocessing helper: 0 auto (default), 1 native, 2 small */
    int video_max_pending;         /* ready windows waiting behind the in-flight one (default 2) */
    shav_result_callback callback; void* user;    /* NULL = queue results for shav_session_poll */
    shav_crop_tap crop_tap; void* crop_tap_user;  /* tests only; NULL in production */
    double audio_wait_s;           /* audio lag rule budget (see WHEN A HOP IS EMITTED): NaN (default) = video_wait_s,
                                      >= 0 seconds, +inf = rule off (a stalled microphone then holds back all hops).
                                      Added in API 2 (appended: earlier fields keep their offsets). */
} shav_session_config;

typedef struct shav_engine_info {
    const char* version;
    const char* audio_mode;        /* "" when no audio model */
    const char* audio_fingerprint; /* SHA-256 of the audio weights */
    const char* audio_revision;    /* shield_compute_revision() */
    const char* video_backend;     /* "cuda" | "cpu" | "none" */
    const char* video_mode;
    const char* video_fingerprint; /* SHA-256 of the video weights */
    const char* video_revision;    /* CUDA compute revision or CPU revision */
    double video_threshold;
    const char* load_log;          /* what was tried at load (fallbacks) */
} shav_engine_info;

typedef struct shav_engine_stats {
    int64_t sessions;
    int64_t audio_jobs, video_jobs, video_frames_encoded, video_frames_reused, video_dropped;
    int64_t audio_queue_depth, video_queue_depth;     /* current */
    size_t crop_buffers, crop_buffer_bytes;           /* allocated crop staging buffers (pinned on CUDA) */
    size_t video_device_bytes;                        /* CUDA: device memory of the model incl. token slots */
} shav_engine_stats;

typedef struct shav_session_stats {
    int64_t hops_emitted, audio_hops, video_windows_resolved, video_windows_scored;
    int64_t video_frames_pushed, video_frames_sampled, video_frames_encoded, video_frames_reused;
    size_t bytes_total;            /* all of the below */
    size_t bytes_audio;            /* shield_stream_bytes + hop buffer */
    size_t bytes_video_frames;     /* preprocessing helper: cached sampled frames */
    size_t bytes_crops;            /* crop staging buffers currently held */
    size_t bytes_tokens_host;      /* CPU backend token cache (24 x 24,576 B) */
    size_t bytes_tokens_device;    /* CUDA backend token slots (24 x 24,576 B, device memory) */
    size_t bytes_other;            /* session object, pending results */
} shav_session_stats;

void shav_engine_default_config(shav_engine_config* cfg);
shav_engine* shav_engine_create(const shav_engine_config* cfg);
void shav_engine_free(shav_engine* e);   /* all sessions must be freed first */
int shav_engine_get_info(const shav_engine* e, shav_engine_info* info);   /* strings live as long as the engine */
int shav_engine_get_stats(const shav_engine* e, shav_engine_stats* st);
/* Batch predict through the engine's video backend and mode (tests: live path == batch path).
 * clips: b x 20 x 180 x 320 x 3 uint8.  logits: b. */
int shav_engine_video_predict(shav_engine* e, const uint8_t* clips, size_t b, float* logits);
const char* shav_last_error(void);
const char* shav_version(void);

void shav_session_default_config(shav_session_config* cfg);
shav_session* shav_session_create(shav_engine* e, const shav_session_config* cfg);
void shav_session_free(shav_session* s);          /* waits for the session's jobs; undelivered results are dropped */
/* Waits for the session's jobs, drops pending/undelivered results and restarts it as a fresh session (same id and
 * config): hop numbering, the audio ring/low-pass state, the video timeline, the token cache and alert histories
 * restart.  Timestamps pushed afterwards are on the new session clock (0 = next audio sample). */
int shav_session_reset(shav_session* s);
/* Finite float32 PCM in [-1, 1], 16 kHz mono; any number of samples.  Invalid input is rejected whole. */
int shav_session_push_audio(shav_session* s, const float* pcm, size_t samples);
int shav_session_push_audio_s16(shav_session* s, const int16_t* pcm, size_t samples);   /* x / 32768 */
/* One decoded frame, presentation order, t = session seconds (>= 0 unless reference mode), strictly increasing.
 * RGB24 interleaved, or I420 (BT.601 limited range; converted exactly like ffmpeg 6.1 swscale). */
int shav_session_push_video_rgb(shav_session* s, double t, const uint8_t* rgb, int width, int height, ptrdiff_t stride);
int shav_session_push_video_i420(shav_session* s, double t, const uint8_t* y, ptrdiff_t y_stride, const uint8_t* u,
                                 ptrdiff_t u_stride, const uint8_t* v, ptrdiff_t v_stride, int width, int height);
/* End of input: settle every hop whose audio part is complete, wait for all jobs, emit.  Further pushes fail until
 * reset.  timeout_s < 0 = wait forever.  Returns 0, or -1 on timeout. */
int shav_session_finish(shav_session* s, double timeout_s);
/* Wait until the session has no job in flight or pending (does not end input).  0, or -1 on timeout. */
int shav_session_wait_idle(shav_session* s, double timeout_s);
/* Non-blocking: move up to cap emitted results (oldest first) into out; returns the count. */
size_t shav_session_poll(shav_session* s, shav_hop_result* out, size_t cap);
const char* shav_session_id(const shav_session* s);
int shav_session_get_stats(const shav_session* s, shav_session_stats* st);
size_t shav_session_bytes(const shav_session* s);
/* Session-aligned mode: the hop of participant window 0 once the first frame was pushed, else -1. */
int64_t shav_session_video_first_hop(const shav_session* s);

const char* shav_status_name(int status);
const char* shav_reason_name(int reason);
double shav_monotonic(void);              /* CLOCK_MONOTONIC seconds (same clock as the t_* fields) */
/* One JSON object (no newline) for a hop result; NaN as null.  Returns the length that was needed (like snprintf). */
int shav_result_json(const shav_hop_result* r, const char* participant_id, char* buf, size_t cap);

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
#ifdef __cplusplus
}
#endif
#endif /* SHIELD_AV_H */
