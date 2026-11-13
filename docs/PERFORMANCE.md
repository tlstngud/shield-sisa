# Performance

Hardware: AMD Ryzen 9 9950X3D (16 cores / 32 threads) and NVIDIA RTX A6000 (48 GB). The machine had other load during the runs. "PyTorch" is the original Python/PyTorch pipeline, measured on the same machine.

## Video preprocessing (one 10 s window, decode excluded)

| | Python | native |
|---|---:|---:|
| whole window | 302 ms | 32.5 ms (AVX-512) |
| Haar face detection, per anchor frame | 23.6 ms | 3.95 ms |
| live cost per participant-second | 163.5 ms | 15.5 ms |

The output is bit-identical to the Python reference on all 840 RTFS val+test videos, with both RGB24 and I420 input, at the scalar, AVX2 and AVX-512 dispatch levels.

## SHIELD-V model on CPU (per clip, 20 frames)

| | PyTorch FP32 | native exact | native BF16 |
|---|---:|---:|---:|
| 1 thread | 263 ms | 59.0 ms | 40.3 ms |
| 8 threads | ~118 ms | 8.2 ms | 5.8 ms |
| live update (4 new frames), 8 threads | — | 3.4 ms | 2.5 ms |

## SHIELD-V model on GPU

| batch | PyTorch BF16 | native FP16 |
|---:|---:|---:|
| 1 | 5.41 ms | 2.89 ms |
| 4 | 13.4 ms | 8.04 ms |
| 32 | 108 ms | 51.9 ms |
| 128 | 505 ms | 191 ms |
| 512 | out of memory | 771 ms (664 clips/s) |

- Model-only throughput: the CNN runs at 17,100 frames/s and the sequence stages at 5,500 windows/s.
- 512 participants updating at once:
  - with 4 new frames each, p99 latency is 221 ms;
  - with 20 new frames each, p99 latency is 705 ms.

## Live service (2 s deadline, at most 16 busy CPU threads, 1080p I420 input decoded in memory)

| configuration | audio p99 | video p99 | missed hops |
|---|---:|---:|---:|
| CUDA, N=1 | 51 ms | 16 ms | 0 / 32 |
| CUDA, N=128, burst | 730 ms | 199 ms | 0 / 4,096 |
| CUDA, N=128, staggered | 55 ms | 31 ms | 0 / 4,096 |
| CUDA, N=256, burst, 8 audio workers | 1.12 s | 387 ms | 0 / 8,192 |
| CUDA, N=256, staggered | 203 ms | 42 ms | 0 / 8,192 |
| CUDA, N=384, staggered | 1.22 s | 45 ms | 0 / 12,288 |
| CPU exact, N=128, burst | 1.05 s | 1.56 s | 0 / 4,096 |
| CPU exact, N=128, staggered | 96 ms | 134 ms | 0 / 4,096 |
| video only, CUDA, N=512, staggered | — | 67 ms | 0 |

- CPU cost per participant on the CUDA backend: about 26 ms per second. Of that, preprocessing takes 15.5 ms, audio 9.1 ms and the GPU worker 1.3 ms.
- CPU exact backend: about 59 ms per participant-second for audio and video combined.
- Memory:
  - session state is 46–51 MB per participant;
  - GPU memory is about 7 GB with FP16, nearly independent of N.

Each load configuration was run once. RGB24 input, other resolutions, real decoders and network transport were not load-tested.
