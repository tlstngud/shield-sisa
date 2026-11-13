// SHIELD-V CUDA engine: shared device helpers (storage types, conversions, activations).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace shv {

#define SHV_CUDA_CHECK(expr)                                                                         \
    do {                                                                                             \
        cudaError_t err__ = (expr);                                                                  \
        if (err__ != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(err__) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__) + ": " #expr);        \
    } while (0)

// ------------------------------------------------------------------------------------------------ precision
enum class Precision { FP32 = 0, BF16 = 1, FP16 = 2 };

template <typename T> struct StorageTraits;
template <> struct StorageTraits<float> { static constexpr int bytes = 4; static constexpr bool tensor = false; };
template <> struct StorageTraits<__nv_bfloat16> { static constexpr int bytes = 2; static constexpr bool tensor = true; };
template <> struct StorageTraits<__half> { static constexpr int bytes = 2; static constexpr bool tensor = true; };

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }

template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half_rn(x); }

// 8 consecutive elements <-> 8 floats (16-byte loads for half types, 2x float4 for float).
template <typename T> __device__ __forceinline__ void load8(const T* p, float* v);
template <> __device__ __forceinline__ void load8<float>(const float* p, float* v) {
    const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
template <typename H> __device__ __forceinline__ void load8_half(const H* p, float* v) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const H* h = reinterpret_cast<const H*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = to_f(h[i]);
}
template <> __device__ __forceinline__ void load8<__nv_bfloat16>(const __nv_bfloat16* p, float* v) { load8_half(p, v); }
template <> __device__ __forceinline__ void load8<__half>(const __half* p, float* v) { load8_half(p, v); }

template <typename T> __device__ __forceinline__ void store8(T* p, const float* v);
template <> __device__ __forceinline__ void store8<float>(float* p, const float* v) {
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}
template <typename H> __device__ __forceinline__ void store8_half(H* p, const float* v) {
    uint4 u;
    H* h = reinterpret_cast<H*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = from_f<H>(v[i]);
    *reinterpret_cast<uint4*>(p) = u;
}
template <> __device__ __forceinline__ void store8<__nv_bfloat16>(__nv_bfloat16* p, const float* v) { store8_half(p, v); }
template <> __device__ __forceinline__ void store8<__half>(__half* p, const float* v) { store8_half(p, v); }

// ------------------------------------------------------------------------------------------------ activations
// Same formulas as the ATen CPU kernels (x * relu6(x + 3) / 6 etc.); no fast-math anywhere in this engine.
enum Act { ACT_NONE = 0, ACT_RELU = 1, ACT_HSWISH = 2, ACT_SILU = 3, ACT_HSIGMOID = 4 };

__device__ __forceinline__ float relu6p3(float x) { return fminf(fmaxf(x + 3.0f, 0.0f), 6.0f); }
__device__ __forceinline__ float hardswish(float x) { return x * relu6p3(x) / 6.0f; }
__device__ __forceinline__ float hardsigmoid(float x) { return relu6p3(x) / 6.0f; }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
// torch log_sigmoid: min(x, 0) - log1p(exp(-|x|))
__device__ __forceinline__ float logsigmoidf_(float x) { return fminf(x, 0.0f) - log1pf(expf(-fabsf(x))); }

__device__ __forceinline__ float apply_act(float v, int act) {
    switch (act) {
    case ACT_RELU: return fmaxf(v, 0.0f);
    case ACT_HSWISH: return hardswish(v);
    case ACT_SILU: return silu(v);
    case ACT_HSIGMOID: return hardsigmoid(v);
    default: return v;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

inline int ceil_div(long long a, long long b) { return int((a + b - 1) / b); }

}  // namespace shv
