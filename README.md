# SHIELD-SISA

Native C++/CUDA engine and live service for real-time **audio + video deepfake detection** in calls and meetings.
It runs two models built on SISA sequence layers:

- **SHIELD-A**: the audio model. It reads 10 s of 16 kHz speech.
- **SHIELD-V**: the video model. It uses a MobileNetV3 frame encoder, then GDN and SISA sequence stages, over 20 face-crop frames sampled at 2 fps across a 10 s window.

The service keeps one session per participant. It takes PCM audio and video frames at the same time and reports an audio verdict, a video verdict and a combined alert every 2 s.

> Demo / research release (non-commercial demo day). The model weights are published separately on Hugging Face under CC BY-NC 4.0: <https://huggingface.co/koreashin/shield-sisa>. This repository (code) is Apache-2.0.

## What is in here

| Path | What |
|---|---|
| `engine/audio` | SHIELD-A native engine (C++17, AVX2/AVX-512 runtime dispatch), with a streaming API and an optional low-pass filter |
| `engine/preprocess` | SHIELD-V preprocessing in C++: 0.5 s frame sampling, Pillow-exact bilinear letterbox, OpenCV-exact gray conversion and Haar face detector, crop interpolation, area resize. Its output is bit-identical to the Python reference. |
| `engine/model` | SHIELD-V CPU engine: exact FP32 plus certified BF16 modes |
| `engine/cuda` | SHIELD-V CUDA engine (sm_80+): FP32 exact plus certified FP16 and BF16, with an async batched API |
| `service` | `libshield_av`, the integrated live audio+video session API (C ABI) |
| `tools` | `demo_av` (runs one participant from a file), the preprocessing CLI, and the weight export |
| `bench` | contract, edge-case, parity and load tests; certification and metric tools |
| `oracle` | pure-PyTorch float32/float64 reference of SHIELD-V, used as ground truth |

## Build

Requirements:
- Linux x86-64
- GCC 11+
- CMake 3.16+
- (optional) Ninja
- `ffmpeg` and `ffprobe` on `PATH`, for the file demo and tests
- CUDA 12+ for the GPU engine

```bash
./build.sh                         # -> build/{audio,prep,cpu,cuda,service}
CUDA_ARCH=89 ./build.sh            # pick the GPU architecture (default 86; sm_80+ required)
```

Download the weights from Hugging Face into `build/`:

- `shield_v.weights`
- `shield_v.weights.json`
- `shield_v.weights.cpu.cert`
- `shield_v.weights.cuda.cert`
- `audio/shield.weights`
- `audio/shield.weights.json`

```bash
hf download koreashin/shield-sisa --local-dir build
```

## Quick start (C API)

```c
#include "shield_av.h"

shav_engine_config ec; shav_engine_default_config(&ec);
ec.audio_weights = "build/audio/shield.weights";
ec.video_weights = "build/shield_v.weights";
ec.video_backend = "cpu";            // "auto" (CUDA if available), "cuda", "cpu", "none"
shav_engine* eng = shav_engine_create(&ec);

shav_session_config sc; shav_session_default_config(&sc);
sc.participant_id = "alice";
shav_session* s = shav_session_create(eng, &sc);

// from your capture threads (audio and video of one session may be pushed concurrently)
shav_session_push_audio(s, pcm, n_samples);                       // float32 [-1,1], 16 kHz mono
shav_session_push_video_rgb(s, t_seconds, rgb, width, height, stride);   // or _i420
shav_hop_result r[8];
size_t n = shav_session_poll(s, r, 8);                            // non-blocking; or set sc.callback
for (size_t i = 0; i < n; i++) { char buf[2048]; shav_result_json(&r[i], "alice", buf, sizeof buf); puts(buf); }

shav_session_finish(s, -1); shav_session_free(s); shav_engine_free(eng);
```

Try it on a file with `build/service/demo_av --video clip.mp4 --audio clip.mp4 --backend cpu`. It prints one JSON line per hop, and ffmpeg decodes both streams.

The full contract is in [`service/include/shield_av.h`](service/include/shield_av.h) and [`docs/API.md`](docs/API.md).

### Live contract

- Each hop covers a 10 s window and hops advance every 2 s. Hop *h* covers [2h−8, 2h+2) s.
- Each modality reports one of three statuses: `collecting`, `valid` or `unavailable`. Audio can also report `insufficient_speech`.
- Each status comes with a reason: warmup, no_input, gap, lag, disabled, failed, overload, silence or ended.
- `logit` and `score` are **NaN**, never 0, unless the status is `valid`. A valid video part also reports `face_detection_anchors` (0–5).
- If one modality stalls (muted microphone, frozen camera), the session does not block the other. After a configurable wait (`audio_wait_s`, `video_wait_s`), the stalled part is reported as `unavailable/lag`.

## Quality tiers

The exact FP32 tiers are the default on CPU. On the RTFS val+test split (840 videos), their logits match the float64 PyTorch oracle within 2.8e-4 (CPU) and 4.0e-4 (CUDA), with no flipped decisions.

| | test AUROC | false alarms (Real) | detected (Fake) |
|---|---|---|---|
| exact FP32 (CPU and CUDA) | 0.9908 | 21 / 232 | 180 / 181 |

Reduced-precision modes are only loaded when a certificate binds them to the exact weight SHA-256, the engine revision and the decision threshold. A mode is certified only if all of these hold against exact FP32, per split and per fake method:

- AUROC drops by at most 0.002;
- the Real false-positive rate rises by at most 1 pp;
- the Fake true-positive rate falls by at most 1 pp.

Certified modes: CUDA FP16 and BF16; CPU BF16-CNN, BF16-seq and BF16. CPU INT8 failed certification and is refused.

## Performance

Hardware: AMD Ryzen 9 9950X3D (16 cores), NVIDIA RTX A6000. "PyTorch" is the original Python/PyTorch pipeline on the same machine.

| | PyTorch | SHIELD-SISA | speed-up |
|---|---:|---:|---:|
| Video preprocessing, one 10 s window | 302 ms | 32.5 ms | 9.3× |
| Video preprocessing, CPU per participant-second (live) | 163.5 ms | 15.5 ms | 10.5× |
| SHIELD-V on CPU, per clip, 1 thread | 263 ms | 59 ms | 4.5× |
| SHIELD-V on CPU, per clip, 8 threads | ~118 ms | 8.2 ms | 14× |
| SHIELD-V on GPU, batch 1 | 5.41 ms | 2.89 ms (FP16) | 1.9× |
| SHIELD-V on GPU, batch 128 | 505 ms | 191 ms | 2.6× |
| SHIELD-V on GPU, peak throughput | 296 clips/s | 664 clips/s | 2.2× |
| SHIELD-V on GPU, batch 512 | out of memory | 771 ms | — |

Live service load (2 s deadline, 16 CPU threads, one GPU, 1080p input):

| configuration | result |
|---|---|
| CPU-only exact, 128 participants (audio+video) | 0 missed hops, about 8 cores |
| CUDA, 256 participants, burst arrivals (audio+video) | 0 missed hops |
| CUDA, 384 participants, staggered arrivals (audio+video) | 0 missed hops |
| CUDA, 512 participants, video only | 0 missed hops |

Memory:
- About 50 MB of host memory per participant (the frame store of one 10 s window).
- About 7 GB of GPU memory with FP16, nearly independent of the number of participants.
- 40 MB of video weights.

### Deployment note (e.g. AWS)

- The **CPU-only exact path** is the recommended deployment for now. It uses about 0.06 core per participant-second for audio+video combined.
- Use x86-64 instances with AVX-512, such as `c7i` or `c7a`. ARM (Graviton) is not supported.

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) and [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## 한국어 요약

SHIELD-SISA는 통화·회의에서 음성(SHIELD-A)과 영상(SHIELD-V) 딥페이크를 실시간으로 함께 탐지하는 C++/CUDA 네이티브 엔진과 라이브 서비스입니다.

- **결과 일치**: 원래 Python/PyTorch 파이프라인과 비교했습니다.
  - 전처리는 비트 단위로 같은 결과를 냅니다.
  - CPU 정확 모드는 PyTorch FP32와 같은 품질입니다(test AUROC 0.9908).
- **속도**: 전처리 약 10배, CPU 모델 4.5~14배, GPU 모델 약 2~2.6배 빨라졌습니다.
- **서비스**: 참가자별 세션이 음성과 영상을 동시에 받아 2초마다 판정합니다.
- **배포**: 지금은 CPU 단독 배포를 권장합니다. AVX-512를 지원하는 x86 인스턴스(AWS c7i/c7a)를 쓰세요.
- **라이선스**: 코드는 Apache-2.0, 가중치는 Hugging Face에서 CC BY-NC 4.0(비상업·데모용)입니다.

## License

Code: Apache License 2.0 (see `LICENSE` and `NOTICE`). Model weights: CC BY-NC 4.0, distributed separately.
