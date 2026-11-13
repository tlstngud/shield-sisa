# libshield_av API

The header is the authoritative contract: [`service/include/shield_av.h`](../service/include/shield_av.h). This page summarises it.

## Engine

1. Call `shav_engine_default_config(&c)`.
2. Set the weight paths, `audio_weights` and `video_weights`.
3. Set `video_backend`: `auto`, `cuda`, `cpu` or `none`.
4. Optionally set `video_mode`:
   - `NULL` means CUDA FP16, falling back to FP32 when FP16 has no valid certificate, or CPU exact;
   - an explicit mode never falls back.

Sizing options: `audio_workers` (1–8), `max_sessions`, `cpu_video_workers`.

Lifecycle: `shav_engine_create`, `shav_engine_get_info`, `shav_engine_get_stats`, `shav_engine_free`.

## Session (one per participant)

1. Call `shav_session_default_config(&s)`. `participant_id` is required.
2. Optional settings:
   - `policy`: `EITHER` (default), `BOTH`, `AUDIO` or `VIDEO`
   - `audio_threshold`
   - `audio_alert_k/n` and `video_alert_k/n` (k-of-n alerting)
   - `audio_lowpass` (default 1)
   - `video_wait_s` and `audio_wait_s` (stall waits)
   - `video_max_pending` and `video_cache`
   - `callback` + `user`, or poll instead

### Input

- Audio: `shav_session_push_audio` (float32) or `shav_session_push_audio_s16`, 16 kHz mono.
- Video: `shav_session_push_video_rgb` or `shav_session_push_video_i420`, one decoded frame per call. Pass the session time `t` in seconds; it must strictly increase.
- Threading:
  - Audio and video of one session may be pushed from two threads at once.
  - No other calls on the same session may overlap.
  - Video preprocessing runs on the pushing thread, so that thread needs its own CPU.

### Output

- Delivery:
  - `shav_session_poll(s, out, cap)` is non-blocking;
  - alternatively, use the callback.
- `shav_result_json` serialises a result, writing NaN as `null`.

### Ending a session

- `shav_session_finish(s, timeout_s)`
- `shav_session_wait_idle`
- `shav_session_reset`
- `shav_session_free`

## Results

- Hop *h* ends at 2(h+1) s and covers [2h−8, 2h+2) s. Hops are emitted in order.
- `status` is one of `collecting`, `valid`, `unavailable`, or (audio only) `insufficient_speech`.
- `reason` is one of `warmup`, `no_input`, `gap`, `lag`, `disabled`, `failed`, `overload`, `silence`, `ended`.
- Unless a part is `valid`:
  - `logit` and `score` are NaN;
  - `decision` and `alert` are −1.
- A valid video part also reports `face_detection_anchors` (0–5).
- Timing fields: `t_input`, `t_done`, `queue_ms`, `compute_ms`, `batch_windows`, `frames_encoded`, `frames_reused`.

## Backpressure

- Each participant has at most one job in flight per modality.
- Video windows beyond `video_max_pending` are dropped as `unavailable/overload`.
- Audio hops are never dropped.
