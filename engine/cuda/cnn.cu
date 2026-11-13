// SHIELD-V CUDA engine: MobileNetV3 kernels other than the 1x1 convolutions (those are GEMMs, gemm.cuh).
#include "kernels.cuh"

namespace shv {
namespace {

// ------------------------------------------------------------------------------------------------ stem
template <typename T>
__global__ void __launch_bounds__(256)
stem_kernel(const uint8_t* __restrict__ in, T* __restrict__ out, const float* __restrict__ lut,
            const float* __restrict__ w, const float* __restrict__ b, int F) {
    __shared__ float slut[768];
    __shared__ __align__(16) float sw[27 * 16];
    __shared__ float sb[16];
    for (int i = threadIdx.x; i < 768; i += blockDim.x) slut[i] = lut[i];
    for (int i = threadIdx.x; i < 27 * 16; i += blockDim.x) sw[i] = w[i];
    if (threadIdx.x < 16) sb[threadIdx.x] = b[threadIdx.x];
    __syncthreads();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)F * 90 * 160) return;
    const int ow = int(idx % 160), oh = int((idx / 160) % 90);
    const long long f = idx / (90 * 160);
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        const int ih = 2 * oh - 1 + kh;
        if (ih < 0 || ih >= IMG_H) continue;
#pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            const int iw = 2 * ow - 1 + kw;
            if (iw < 0 || iw >= IMG_W) continue;
            const uint8_t* px = in + ((f * IMG_H + ih) * IMG_W + iw) * 3;
#pragma unroll
            for (int ci = 0; ci < 3; ++ci) {
                const float x = slut[ci * 256 + px[ci]];
                const float* wr = sw + ((kh * 3 + kw) * 3 + ci) * 16;
#pragma unroll
                for (int co = 0; co < 16; ++co) acc[co] = fmaf(wr[co], x, acc[co]);
            }
        }
    }
    float v[16];
#pragma unroll
    for (int co = 0; co < 16; ++co) v[co] = hardswish(acc[co] + sb[co]);
    store8(out + idx * 16, v);
    store8(out + idx * 16 + 8, v + 8);
}

// ------------------------------------------------------------------------------------------------ depthwise
template <typename T, int KS, int S>
__global__ void __launch_bounds__(256)
dw_kernel(const T* __restrict__ in, T* __restrict__ out, const float* __restrict__ w, const float* __restrict__ b,
          int F, int H, int W, int C, int Ho, int Wo, int act) {
    const int CG = C >> 3;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)F * Ho * Wo * CG) return;
    const int cg = int(idx % CG);
    const long long pix = idx / CG;
    const int ow = int(pix % Wo);
    const long long r2 = pix / Wo;
    const int oh = int(r2 % Ho);
    const long long f = r2 / Ho;
    constexpr int P = (KS - 1) / 2;
    const int c0 = cg * 8;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int kh = 0; kh < KS; ++kh) {
        const int ih = oh * S - P + kh;
        if (ih < 0 || ih >= H) continue;
#pragma unroll
        for (int kw = 0; kw < KS; ++kw) {
            const int iw = ow * S - P + kw;
            if (iw < 0 || iw >= W) continue;
            float x[8];
            load8(in + ((f * H + ih) * W + iw) * C + c0, x);
            const float4 w0 = __ldg(reinterpret_cast<const float4*>(w + (kh * KS + kw) * C + c0));
            const float4 w1 = __ldg(reinterpret_cast<const float4*>(w + (kh * KS + kw) * C + c0 + 4));
            acc[0] = fmaf(w0.x, x[0], acc[0]); acc[1] = fmaf(w0.y, x[1], acc[1]);
            acc[2] = fmaf(w0.z, x[2], acc[2]); acc[3] = fmaf(w0.w, x[3], acc[3]);
            acc[4] = fmaf(w1.x, x[4], acc[4]); acc[5] = fmaf(w1.y, x[5], acc[5]);
            acc[6] = fmaf(w1.z, x[6], acc[6]); acc[7] = fmaf(w1.w, x[7], acc[7]);
        }
    }
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = apply_act(acc[i] + b[c0 + i], act);
    store8(out + pix * C + c0, v);
}

// ------------------------------------------------------------------------------------------------ SE pooling
// Two deterministic stages with a split that depends only on the layer (never on the batch):
//   part[f][s][c] = sum over hw in [64 s, 64 s + 64) (8 strided row groups, combined in order 0..7)
//   pooled[f][c]  = (sum over s = 0..S-1 of part[f][s][c]) / HW
constexpr int SE_CHUNK = 64;
template <typename T>
__global__ void __launch_bounds__(256)
se_part_kernel(const T* __restrict__ x, float* __restrict__ part, int HW, int C, int S) {
    const int f = blockIdx.x, lane = threadIdx.x & 31, rg = threadIdx.x >> 5, c = blockIdx.y * 32 + lane, sp = blockIdx.z;
    const int hw0 = sp * SE_CHUNK, hw1 = min(HW, hw0 + SE_CHUNK);
    float acc = 0.0f;
    if (c < C)
        for (int hw = hw0 + rg; hw < hw1; hw += 8) acc += to_f(x[((long long)f * HW + hw) * C + c]);
    __shared__ float red[8][32];
    red[rg][lane] = acc;
    __syncthreads();
    if (rg == 0 && c < C) {
        float t = red[0][lane];
#pragma unroll
        for (int r = 1; r < 8; ++r) t += red[r][lane];
        part[((long long)f * S + sp) * C + c] = t;
    }
}
__global__ void se_finalize_kernel(const float* __restrict__ part, float* __restrict__ pooled, int F, int S, int C, int HW) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)F * C) return;
    const long long f = i / C;
    const int c = int(i % C);
    float t = 0.0f;
    for (int sp = 0; sp < S; ++sp) t += part[(f * S + sp) * C + c];
    pooled[i] = t / float(HW);
}

// ------------------------------------------------------------------------------------------------ 4x4 pool
template <typename T>
__global__ void __launch_bounds__(256) pool_kernel(const T* __restrict__ in, T* __restrict__ out, int F) {
    constexpr int CG = CNN_OUT / 8;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)F * 16 * CG) return;
    const int cg = int(idx % CG), tok = int((idx / CG) % 16);
    const long long f = idx / (16 * CG);
    const int r = tok / 4, c = tok % 4;
    const int rs = (r * 6) / 4, re = ((r + 1) * 6 + 3) / 4, cs = (c * 10) / 4, ce = ((c + 1) * 10 + 3) / 4;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    for (int ih = rs; ih < re; ++ih)
        for (int iw = cs; iw < ce; ++iw) {
            float x[8];
            load8(in + ((f * 6 + ih) * 10 + iw) * CNN_OUT + cg * 8, x);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += x[i];
        }
    const float n = float((re - rs) * (ce - cs));
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = acc[i] / n;
    store8(out + (f * 16 + tok) * CNN_OUT + cg * 8, acc);
}

}  // namespace

template <typename T>
void launch_stem(const uint8_t* in, T* out, const float* lut, const float* w, const float* b, int F, cudaStream_t s) {
    const long long n = (long long)F * 90 * 160;
    stem_kernel<T><<<ceil_div(n, 256), 256, 0, s>>>(in, out, lut, w, b, F);
}

template <typename T>
void launch_dw(const T* in, T* out, const float* w, const float* b, int F, int H, int W, int C, int K, int S, int Ho,
               int Wo, int act, cudaStream_t s) {
    const long long n = (long long)F * Ho * Wo * (C / 8);
    const int grid = ceil_div(n, 256);
    if (K == 3 && S == 1) dw_kernel<T, 3, 1><<<grid, 256, 0, s>>>(in, out, w, b, F, H, W, C, Ho, Wo, act);
    else if (K == 3 && S == 2) dw_kernel<T, 3, 2><<<grid, 256, 0, s>>>(in, out, w, b, F, H, W, C, Ho, Wo, act);
    else if (K == 5 && S == 1) dw_kernel<T, 5, 1><<<grid, 256, 0, s>>>(in, out, w, b, F, H, W, C, Ho, Wo, act);
    else if (K == 5 && S == 2) dw_kernel<T, 5, 2><<<grid, 256, 0, s>>>(in, out, w, b, F, H, W, C, Ho, Wo, act);
    else throw std::runtime_error("unsupported depthwise kernel");
}

int se_splits(int HW) { return (HW + SE_CHUNK - 1) / SE_CHUNK; }
template <typename T>
void launch_se_pool(const T* x, float* part, float* pooled, int F, int HW, int C, cudaStream_t s) {
    const int S = se_splits(HW);
    se_part_kernel<T><<<dim3(F, (C + 31) / 32, S), 256, 0, s>>>(x, part, HW, C, S);
    se_finalize_kernel<<<ceil_div((long long)F * C, 256), 256, 0, s>>>(part, pooled, F, S, C, HW);
}

template <typename T> void launch_pool(const T* in, T* out, int F, cudaStream_t s) {
    const long long n = (long long)F * 16 * (CNN_OUT / 8);
    pool_kernel<T><<<ceil_div(n, 256), 256, 0, s>>>(in, out, F);
}

#define SHV_INST_CNN(T)                                                                                       \
    template void launch_stem<T>(const uint8_t*, T*, const float*, const float*, const float*, int, cudaStream_t); \
    template void launch_dw<T>(const T*, T*, const float*, const float*, int, int, int, int, int, int, int, int, int, \
                               cudaStream_t);                                                                 \
    template void launch_se_pool<T>(const T*, float*, float*, int, int, int, cudaStream_t);                           \
    template void launch_pool<T>(const T*, T*, int, cudaStream_t);
SHV_INST_CNN(float)
SHV_INST_CNN(__nv_bfloat16)
SHV_INST_CNN(__half)

}  // namespace shv
