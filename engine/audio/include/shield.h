#ifndef SHIELD_INFERENCE_H
#define SHIELD_INFERENCE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define SHIELD_WINDOW_SAMPLES 160000
#define SHIELD_HOP_SAMPLES 32000
typedef struct shield_model shield_model;
typedef struct shield_stream shield_stream;
typedef struct shield_ticket shield_ticket;
enum shield_status { SHIELD_COLLECTING=0, SHIELD_INSUFFICIENT_SPEECH=1,
                     SHIELD_VALID=2, SHIELD_UNAVAILABLE=3 };
typedef struct shield_result {
    double window_start, window_end, valid_speech_sec;
    double score; /* Reference service computes sigmoid/threshold in float64. */
    float logit; /* Both score and logit are NaN unless status==SHIELD_VALID. */
    int status, alert; /* alert==-1 means null. */
} shield_result;
/* Immutable shared weights and a bounded persistent worker pool (1..8).
   All audio is finite 16kHz mono float32 PCM in [-1,1]. No resampling.
   The caller owns output storage. Functions return 0 on success, -1 on error.
   Destruction must not race calls on the same handle. */
shield_model* shield_model_load(const char* weights_path, int workers);
/* Exact stays the default. Approximate modes require a passing certificate
   beside the exported file (weights_path + ".cert"), bound to its SHA-256.
   Mode names: exact, fp16-linear, bf16-linear, fp16-cnn, bf16-cnn, int8-cnn. */
shield_model* shield_model_load_mode(const char* weights_path,int workers,const char* mode);
/* Offline certification only; disabled in ordinary builds. */
shield_model* shield_model_load_experimental(const char* weights_path,int workers,const char* mode);
const char* shield_model_mode(const shield_model*);
const char* shield_model_fingerprint(const shield_model*);
const char* shield_compute_revision(void);
void shield_model_free(shield_model*);
const char* shield_last_error(void);
const char* shield_kernel(void);
size_t shield_weight_bytes(const shield_model*);
int shield_predict(shield_model*, const float* pcm, float* logit);
int shield_predict_batch(shield_model*, const float* pcm, size_t count,
                         size_t stride_samples, float* logits);
/* One handle per participant. Stream state is independent; a shared worker pool
   schedules ready windows from all callers FIFO. Calls on one stream serialize.
   A null model is allowed: speech produces unavailable, never a zero score. */
shield_stream* shield_stream_create(shield_model*, const char* participant_id,
    double threshold, const char* version, int alert_k, int alert_n);
void shield_stream_free(shield_stream*);
void shield_stream_reset(shield_stream*);
const char* shield_stream_id(const shield_stream*);
const char* shield_stream_version(const shield_stream*);
size_t shield_stream_bytes(const shield_stream*);
/* Model-input low-pass, off by default. Checkpoints trained on SHIELD's 7.2 kHz low-passed windows need it on.
   Causal: the order-8 Butterworth sections (scipy butter(8, 0.9)) run forward twice, matching the training's
   zero-phase sosfiltfilt in magnitude; the output is clipped to +-32767/32768 like the int16 training windows;
   state carries across pushes and hops and clears on reset. VAD and
   valid_speech_sec still use the raw PCM. Changes only on an empty stream (after create or reset); enabling adds
   a 640,000-byte low-passed ring per participant. shield_stream_lowpass returns 0/1, or -1 for null. */
int shield_stream_set_lowpass(shield_stream*,int enabled);
int shield_stream_lowpass(const shield_stream*);
/* Exact preprocessing cache: 0 off, 1 GDN-A projections, 2 also short-conv
   outputs. Changing level invalidates sequence reuse; frontend policy stays. */
int shield_stream_set_cache_level(shield_stream*,int level);
/* Every completed 2s hop emits a result, including collecting. capacity must
   cover every emitted result; insufficient capacity fails before state changes.
   t_end is the time of the last supplied sample, or NaN for wall clock time.
   Output times correspond to their actual sample positions within the push. */
int shield_stream_push(shield_stream*, const float* pcm, size_t samples,
    double t_end, shield_result* results, size_t capacity, size_t* result_count);
double shield_speech_seconds(const float* pcm, size_t samples);
/* Nonblocking submission for a complete aligned 2-second hop. A participant has
   one inference in flight: another call for that participant applies backpressure.
   Tickets own the model lifetime; freeing a pending ticket waits for completion.
   Other participants need no caller threads while their jobs run in the pool. */
shield_ticket* shield_stream_submit_hop(shield_stream*,const float* pcm_32000,double t_end);
int shield_ticket_ready(shield_ticket*);
int shield_ticket_wait(shield_ticket*,shield_result*);
/* Timing is read only after completion: 0 ready, 1 pending, -1 bad arguments.
   completed_monotonic uses CLOCK_MONOTONIC seconds on Linux. */
int shield_ticket_timing(shield_ticket*,double* queue_ms,double* compute_ms,double* completed_monotonic);
void shield_ticket_free(shield_ticket*);
/* Stable per-ID phase in [0,32000). A capture/arrival scheduler may use this
   to distribute hop boundaries across two seconds. No samples are skipped. */
uint32_t shield_stream_phase_samples(const shield_stream*);
#ifdef __cplusplus
}
#endif
#endif
