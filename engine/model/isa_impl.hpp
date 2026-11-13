// SHIELD-V CPU engine kernels, compiled once per ISA level (isa_avx512.cpp / isa_avx2.cpp / isa_scalar.cpp).
//
// Every kernel is written against a 16-lane float vector "V" whose operations are single IEEE-754 operations
// per lane (add, sub, mul, div, sqrt, fused multiply-add, min, max).  The AVX-512 build maps V to one zmm, the
// AVX2 build to two ymm, the scalar build to float[16] with __builtin_fmaf.  Reductions across lanes use one
// fixed tree (vsum).  Every output element is therefore computed by the same sequence of roundings on every ISA,
// at every tile position, batch size and thread count: results are bit-identical across dispatch levels,
// batches and thread partitions (tested in bench/cpu_verify.py).  Compile with -ffp-contract=off.
//
// ODR safety: this file must not include C++ standard headers nor use std templates (they could be emitted as
// COMDAT functions compiled for AVX-512 and picked by the linker for baseline code).  Everything is internal.
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "net.hpp"

#if SHV_LEVEL >= 2
#include <immintrin.h>
#endif

#define SHV_INL __attribute__((always_inline)) inline

namespace {
using namespace shv;

// ------------------------------------------------------------------------------------------------ vector type
#if SHV_LEVEL == 3
struct V { __m512 v; };
SHV_INL V vload(const float* p) { return V{_mm512_loadu_ps(p)}; }
SHV_INL void vstore(float* p, V a) { _mm512_storeu_ps(p, a.v); }
SHV_INL V vset(float x) { return V{_mm512_set1_ps(x)}; }
SHV_INL V vzero() { return V{_mm512_setzero_ps()}; }
SHV_INL V vadd(V a, V b) { return V{_mm512_add_ps(a.v, b.v)}; }
SHV_INL V vsub(V a, V b) { return V{_mm512_sub_ps(a.v, b.v)}; }
SHV_INL V vmul(V a, V b) { return V{_mm512_mul_ps(a.v, b.v)}; }
SHV_INL V vdiv(V a, V b) { return V{_mm512_div_ps(a.v, b.v)}; }
SHV_INL V vsqrt(V a) { return V{_mm512_sqrt_ps(a.v)}; }
SHV_INL V vfma(V a, V b, V c) { return V{_mm512_fmadd_ps(a.v, b.v, c.v)}; }  // a*b+c, one rounding
SHV_INL V vmax(V a, V b) { return V{_mm512_max_ps(a.v, b.v)}; }               // a>b ? a : b
SHV_INL V vmin(V a, V b) { return V{_mm512_min_ps(a.v, b.v)}; }               // a<b ? a : b
SHV_INL V vpow2i(V n) {  // 2^n for integral n in [-126,127]
    __m512i e = _mm512_add_epi32(_mm512_cvtps_epi32(n.v), _mm512_set1_epi32(127));
    return V{_mm512_castsi512_ps(_mm512_slli_epi32(e, 23))};
}
SHV_INL float vsum(V a) {  // ((l_i + l_{i+8}) + (.. +4)) + (.. +2) + (.. +1)
    __m256 lo = _mm512_castps512_ps256(a.v);
    __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a.v), 1));
    __m256 s8 = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}
#define SHV_NAME "avx512"
#define SHV_MR 6
#define SHV_NV 4
#elif SHV_LEVEL == 2
struct V { __m256 a, b; };
SHV_INL V vload(const float* p) { return V{_mm256_loadu_ps(p), _mm256_loadu_ps(p + 8)}; }
SHV_INL void vstore(float* p, V x) { _mm256_storeu_ps(p, x.a); _mm256_storeu_ps(p + 8, x.b); }
SHV_INL V vset(float x) { __m256 t = _mm256_set1_ps(x); return V{t, t}; }
SHV_INL V vzero() { __m256 t = _mm256_setzero_ps(); return V{t, t}; }
SHV_INL V vadd(V x, V y) { return V{_mm256_add_ps(x.a, y.a), _mm256_add_ps(x.b, y.b)}; }
SHV_INL V vsub(V x, V y) { return V{_mm256_sub_ps(x.a, y.a), _mm256_sub_ps(x.b, y.b)}; }
SHV_INL V vmul(V x, V y) { return V{_mm256_mul_ps(x.a, y.a), _mm256_mul_ps(x.b, y.b)}; }
SHV_INL V vdiv(V x, V y) { return V{_mm256_div_ps(x.a, y.a), _mm256_div_ps(x.b, y.b)}; }
SHV_INL V vsqrt(V x) { return V{_mm256_sqrt_ps(x.a), _mm256_sqrt_ps(x.b)}; }
SHV_INL V vfma(V x, V y, V z) { return V{_mm256_fmadd_ps(x.a, y.a, z.a), _mm256_fmadd_ps(x.b, y.b, z.b)}; }
SHV_INL V vmax(V x, V y) { return V{_mm256_max_ps(x.a, y.a), _mm256_max_ps(x.b, y.b)}; }
SHV_INL V vmin(V x, V y) { return V{_mm256_min_ps(x.a, y.a), _mm256_min_ps(x.b, y.b)}; }
SHV_INL __m256 pow2i8(__m256 n) {
    __m256i e = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127));
    return _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));
}
SHV_INL V vpow2i(V n) { return V{pow2i8(n.a), pow2i8(n.b)}; }
SHV_INL float vsum(V x) {
    __m256 s8 = _mm256_add_ps(x.a, x.b);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}
#define SHV_NAME "avx2"
#define SHV_MR 2
#define SHV_NV 2
#else
struct V { float f[16]; };
SHV_INL V vload(const float* p) { V r; memcpy(r.f, p, sizeof r.f); return r; }
SHV_INL void vstore(float* p, V x) { memcpy(p, x.f, sizeof x.f); }
SHV_INL V vset(float x) { V r; for (int i = 0; i < 16; ++i) r.f[i] = x; return r; }
SHV_INL V vzero() { return vset(0.0f); }
#define SHV_LANES(expr) V r; for (int i = 0; i < 16; ++i) r.f[i] = (expr); return r
SHV_INL V vadd(V x, V y) { SHV_LANES(x.f[i] + y.f[i]); }
SHV_INL V vsub(V x, V y) { SHV_LANES(x.f[i] - y.f[i]); }
SHV_INL V vmul(V x, V y) { SHV_LANES(x.f[i] * y.f[i]); }
SHV_INL V vdiv(V x, V y) { SHV_LANES(x.f[i] / y.f[i]); }
SHV_INL V vsqrt(V x) { SHV_LANES(__builtin_sqrtf(x.f[i])); }
SHV_INL V vfma(V x, V y, V z) { SHV_LANES(__builtin_fmaf(x.f[i], y.f[i], z.f[i])); }
SHV_INL V vmax(V x, V y) { SHV_LANES(x.f[i] > y.f[i] ? x.f[i] : y.f[i]); }
SHV_INL V vmin(V x, V y) { SHV_LANES(x.f[i] < y.f[i] ? x.f[i] : y.f[i]); }
SHV_INL V vpow2i(V n) {
    V r;
    for (int i = 0; i < 16; ++i) {
        uint32_t u = (uint32_t)((int32_t)n.f[i] + 127) << 23;
        memcpy(&r.f[i], &u, 4);
    }
    return r;
}
SHV_INL float vsum(V x) {
    float s8[8], s4[4], s2[2];
    for (int i = 0; i < 8; ++i) s8[i] = x.f[i] + x.f[i + 8];
    for (int i = 0; i < 4; ++i) s4[i] = s8[i] + s8[i + 4];
    for (int i = 0; i < 2; ++i) s2[i] = s4[i] + s4[i + 2];
    return s2[0] + s2[1];
}
#undef SHV_LANES
#define SHV_NAME "scalar"
#define SHV_MR 4
#define SHV_NV 1
#endif

// ------------------------------------------------------------------------------------------------ profiling
// Profile builds only (-DSHV_CPU_PROFILE): per-stage wall time of rank 0 (meaningful for 1-thread runs).
#ifdef SHV_CPU_PROFILE
double g_prof_ns[64];
long g_prof_n[64];
SHV_INL double prof_now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
#define PROF_BEGIN double prof_t0_ = prof_now()
#define PROF_MARK(id)                                     \
    do {                                                  \
        const double prof_t1_ = prof_now();               \
        g_prof_ns[(id)] += prof_t1_ - prof_t0_;           \
        g_prof_n[(id)] += 1;                              \
        prof_t0_ = prof_t1_;                              \
    } while (0)
void prof_get(double* ns, long* n, int len) {
    for (int i = 0; i < len && i < 64; ++i) { ns[i] = g_prof_ns[i]; n[i] = g_prof_n[i]; g_prof_ns[i] = 0; g_prof_n[i] = 0; }
}
#else
#define PROF_BEGIN do {} while (0)
#define PROF_MARK(id) do {} while (0)
void prof_get(double* ns, long* n, int len) { for (int i = 0; i < len; ++i) { ns[i] = 0; n[i] = 0; } }
#endif

SHV_INL int imin(int a, int b) { return a < b ? a : b; }
SHV_INL int imax(int a, int b) { return a > b ? a : b; }
SHV_INL void split(int n, int rank, int size, int* a, int* b) {
    *a = (int)((long long)n * rank / size);
    *b = (int)((long long)n * (rank + 1) / size);
}
SHV_INL void sync(const Team* t) { if (t->size > 1) t->barrier(t->ctx); }

// exp(x), |error| ~1-2 ulp (Cephes expf coefficients), inputs clamped to [-87.33654, 88.37626]
SHV_INL V vexp(V x) {
    x = vmin(vmax(x, vset(-87.33654f)), vset(88.37626f));
    V n = vsub(vfma(x, vset(1.44269504088896341f), vset(12582912.0f)), vset(12582912.0f));
    V r = vfma(n, vset(-0.693359375f), x);
    r = vfma(n, vset(2.12194440e-4f), r);
    V z = vmul(r, r);
    V p = vset(1.9875691500e-4f);
    p = vfma(p, r, vset(1.3981999507e-3f));
    p = vfma(p, r, vset(8.3334519073e-3f));
    p = vfma(p, r, vset(4.1665795894e-2f));
    p = vfma(p, r, vset(1.6666665459e-1f));
    p = vfma(p, r, vset(5.0000001201e-1f));
    p = vfma(p, z, r);
    p = vadd(p, vset(1.0f));
    return vmul(p, vpow2i(n));
}
SHV_INL V vsilu(V x) { return vdiv(x, vadd(vset(1.0f), vexp(vsub(vzero(), x)))); }
SHV_INL V vhswish(V x) {  // torch: x * min(max(x + 3, 0), 6) / 6
    V t = vmin(vmax(vadd(x, vset(3.0f)), vzero()), vset(6.0f));
    return vdiv(vmul(x, t), vset(6.0f));
}
SHV_INL V vhsig(V x) { return vdiv(vmin(vmax(vadd(x, vset(3.0f)), vzero()), vset(6.0f)), vset(6.0f)); }

// scalar helpers (libm; identical on every ISA build)
SHV_INL float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
SHV_INL float logsigmoid_f(float x) { return (x < 0.0f ? x : 0.0f) - log1pf(expf(-fabsf(x))); }
SHV_INL float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// ------------------------------------------------------------------------------------------------ GEMM
// C[m][n] for panels [p0,p1) of L.  acc = sum_k A[m][k]*W[n][k]: K is split into blocks of SHV_KC; inside a block
// one FMA per k in increasing k starting from +0; the block sums are added in block order (total = b0 + b1 + ...).
// (PyTorch/MKL float32 GEMM accumulates in K blocks of 192; blocks of 64 are ~1.4x more accurate than that and
// ~2.5x more accurate than one sequential chain.)  The epilogue is applied per 16-lane vector.  For EPI_SWIGLU
// panels come in (gate, up) pairs and output pair q is written to C column 16*q.
#define SHV_KC 64
// BF = false: A float [MR][lda], W float panels, k in elements.
// BF = true (AVX-512 BF16 builds only): A = bf16 pairs (uint32) [MR][lda pairs], W = bf16 pair panels
// [KP][16][2], k in pairs; one VDPBF16PS per pair (acc += a1*w1 then += a0*w0, fp32 accumulation).
template <int MR, int NV, bool BF>
SHV_INL void tile_block(const void* Av, int lda, int k0, int k1, const void* Wv, size_t ps, V (&blk)[MR][NV]) {
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i)
#pragma GCC unroll 4
        for (int j = 0; j < NV; ++j) blk[i][j] = vzero();
    if constexpr (!BF) {
        const float* A = static_cast<const float*>(Av);
        const float* W = static_cast<const float*>(Wv);
        for (int k = k0; k < k1; ++k) {
            V w[NV];
#pragma GCC unroll 4
            for (int j = 0; j < NV; ++j) w[j] = vload(W + j * ps + (size_t)k * 16);
#pragma GCC unroll 8
            for (int i = 0; i < MR; ++i) {
                V a = vset(A[(size_t)i * lda + k]);
#pragma GCC unroll 4
                for (int j = 0; j < NV; ++j) blk[i][j] = vfma(a, w[j], blk[i][j]);
            }
        }
    } else {
#if SHV_LEVEL == 3
        const uint32_t* A = static_cast<const uint32_t*>(Av);
        const uint16_t* W = static_cast<const uint16_t*>(Wv);
        for (int k = k0; k < k1; ++k) {
            __m512bh w[NV];
#pragma GCC unroll 4
            for (int j = 0; j < NV; ++j) w[j] = (__m512bh)_mm512_loadu_si512(W + j * ps + (size_t)k * 32);
#pragma GCC unroll 8
            for (int i = 0; i < MR; ++i) {
                const __m512bh a = (__m512bh)_mm512_set1_epi32((int)A[(size_t)i * lda + k]);
#pragma GCC unroll 4
                for (int j = 0; j < NV; ++j) blk[i][j].v = _mm512_dpbf16_ps(blk[i][j].v, a, w[j]);
            }
        }
#else
        (void)Av; (void)Wv; (void)lda; (void)k0; (void)k1; (void)ps;
#endif
    }
}
// Output row mapping of a GEMM: row m -> base + (m / w) * rowstride + (m % w) * ldc (w == 0: base + m * ldc).
// Lets one GEMM write a whole feature map into the interior of a zero-bordered (padded) buffer.
struct OutMap {
    float* base;
    int ldc;
    int w;
    size_t rowstride;
};
SHV_INL float* out_row(const OutMap& o, int m) {
    return o.w ? o.base + (size_t)(m / o.w) * o.rowstride + (size_t)(m % o.w) * o.ldc : o.base + (size_t)m * o.ldc;
}

// register-tile rows per panel count (accumulators: MR*NV vectors)
#if SHV_LEVEL == 3
// measured (scratch/cpu_model/tile_variants.sh): 6/6/8/6 beats 12/12/8/6 by 7% and 8/8/8/6 by 1.3% per frame
#ifndef SHV_MR1
#define SHV_MR1 6
#endif
#ifndef SHV_MR2
#define SHV_MR2 6
#endif
#ifndef SHV_MR3
#define SHV_MR3 8
#endif
template <int NV> struct MRof {
    static constexpr int v = NV == 1 ? SHV_MR1 : NV == 2 ? SHV_MR2 : NV == 3 ? SHV_MR3 : 6;
};
#elif SHV_LEVEL == 2
template <int NV> struct MRof { static constexpr int v = NV == 1 ? 4 : 2; };
#else
template <int NV> struct MRof { static constexpr int v = 4; };
#endif

// INT8 dequantisation data of one tile: per output column (panel-offset) correction 128*sum(w) and weight scale,
// per row activation scale.
struct Q8 {
    const int32_t* corr;
    const float* ws;
    const float* as;
};

// F: 0 = FP32, 1 = BF16 pairs, 2 = INT8 groups of 4 (u8 activation + 128 zero point, s8 weights, VPDPBUSD exact
// int32 accumulation, then (float)(dot - 128*sum w) * (a_scale * w_scale)).
template <int MR, int NV, int F = 0>
SHV_INL void gemm_tile(const void* A, int lda, int K, const void* W, size_t ps, const float* bias, const OutMap& om,
                       int m0, int coff, int epi, const float* R, int ldr, const Q8* q = nullptr) {
    constexpr bool BF = F == 1;
    constexpr int KB = BF ? SHV_KC / 2 : SHV_KC;  // K block (64 products either way)
    V acc[MR][NV];
    if constexpr (F == 2) {
#if SHV_LEVEL == 3
        const uint32_t* a8 = static_cast<const uint32_t*>(A);
        const int8_t* w8 = static_cast<const int8_t*>(W);
        __m512i ai[MR][NV];
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i)
#pragma GCC unroll 4
            for (int j = 0; j < NV; ++j) ai[i][j] = _mm512_setzero_si512();
        for (int k = 0; k < K; ++k) {
            __m512i w[NV];
#pragma GCC unroll 4
            for (int j = 0; j < NV; ++j) w[j] = _mm512_loadu_si512(w8 + j * ps + (size_t)k * 64);
#pragma GCC unroll 8
            for (int i = 0; i < MR; ++i) {
                const __m512i a = _mm512_set1_epi32((int)a8[(size_t)i * lda + k]);
#pragma GCC unroll 4
                for (int j = 0; j < NV; ++j) ai[i][j] = _mm512_dpbusd_epi32(ai[i][j], a, w[j]);
            }
        }
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            const __m512 as = _mm512_set1_ps(q->as[i]);
#pragma GCC unroll 4
            for (int j = 0; j < NV; ++j) {
                const __m512i c = _mm512_loadu_si512(q->corr + j * 16);
                acc[i][j].v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(ai[i][j], c)),
                                            _mm512_mul_ps(as, _mm512_loadu_ps(q->ws + j * 16)));
            }
        }
#else
        (void)q;
        return;
#endif
    } else if (K <= KB) {
        tile_block<MR, NV, BF>(A, lda, 0, K, W, ps, acc);
    } else {
        // block totals live in an L1-resident stack tile; the register tile accumulates one block at a time
        alignas(64) float tot[MR][NV][16];
        tile_block<MR, NV, BF>(A, lda, 0, KB, W, ps, acc);
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i)
#pragma GCC unroll 4
            for (int j = 0; j < NV; ++j) vstore(tot[i][j], acc[i][j]);
        for (int k0 = KB;; k0 += KB) {
            tile_block<MR, NV, BF>(A, lda, k0, imin(K, k0 + KB), W, ps, acc);
            if (k0 + KB >= K) break;  // last block: acc stays in registers and is added below
#pragma GCC unroll 8
            for (int i = 0; i < MR; ++i)
#pragma GCC unroll 4
                for (int j = 0; j < NV; ++j) vstore(tot[i][j], vadd(vload(tot[i][j]), acc[i][j]));
        }
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i)
#pragma GCC unroll 4
            for (int j = 0; j < NV; ++j) acc[i][j] = vadd(vload(tot[i][j]), acc[i][j]);
    }
    if (epi == EPI_SWIGLU) {
#pragma GCC unroll 12
        for (int i = 0; i < MR; ++i) {
            float* crow = out_row(om, m0 + i) + coff;
#pragma GCC unroll 2
            for (int j = 0; j + 1 < NV; j += 2) vstore(crow + (j / 2) * 16, vmul(vsilu(acc[i][j]), acc[i][j + 1]));
        }
        return;
    }
#pragma GCC unroll 12
    for (int i = 0; i < MR; ++i) {
        float* crow = out_row(om, m0 + i) + coff;
#pragma GCC unroll 4
        for (int j = 0; j < NV; ++j) {
            V v = acc[i][j];
            float* c = crow + j * 16;
            switch (epi) {
            case EPI_STORE: if (bias) v = vadd(v, vload(bias + j * 16)); break;
            case EPI_RELU: v = vmax(vadd(v, vload(bias + j * 16)), vzero()); break;
            case EPI_HS: v = vhswish(vadd(v, vload(bias + j * 16))); break;
            case EPI_HSIG: v = vhsig(vadd(v, vload(bias + j * 16))); break;
            case EPI_SILU: v = vsilu(vadd(v, vload(bias + j * 16))); break;
            case EPI_RES: v = vadd(vadd(v, vload(bias + j * 16)), vload(R + (size_t)i * ldr + j * 16)); break;
            case EPI_ACC: v = vadd(vload(c), v); break;
            case EPI_ACC_DIV2: v = vadd(vload(c), vdiv(v, vset(1.41421356237309515f))); break;
            default: break;
            }
            vstore(c, v);
        }
    }
}

template <int NV, int R, int F>
SHV_INL void gemm_tail(int rest, const void* A, int lda, int K, const void* W, size_t ps, const float* bias,
                       const OutMap& om, int m0, int coff, int epi, const float* Rs, int ldr, const Q8* q) {
    if constexpr (R > 0) {
        if (rest == R) gemm_tile<R, NV, F>(A, lda, K, W, ps, bias, om, m0, coff, epi, Rs, ldr, q);
        else gemm_tail<NV, R - 1, F>(rest, A, lda, K, W, ps, bias, om, m0, coff, epi, Rs, ldr, q);
    }
}
// rows [0, M) of A -> output rows [mbase, mbase + M) of om
template <int NV, int F>
void gemm_rows(const void* Av, int lda, int M, int K, const void* W, size_t ps, const float* bias, const OutMap& om,
               int mbase, int coff, int epi, const float* R, int ldr, const Q8* q) {
    constexpr int MR = MRof<NV>::v;
    const char* A = static_cast<const char*>(Av);
    const size_t es = F ? sizeof(uint32_t) : sizeof(float);
    int m = 0;
    Q8 qt{};
    if (q) qt = *q;
    for (; m + MR <= M; m += MR) {
        if (q) qt.as = q->as + m;
        gemm_tile<MR, NV, F>(A + (size_t)m * lda * es, lda, K, W, ps, bias, om, mbase + m, coff, epi,
                             R ? R + (size_t)m * ldr : nullptr, ldr, &qt);
    }
    if (M > m) {
        if (q) qt.as = q->as + m;
        gemm_tail<NV, MR - 1, F>(M - m, A + (size_t)m * lda * es, lda, K, W, ps, bias, om, mbase + m, coff, epi,
                                 R ? R + (size_t)m * ldr : nullptr, ldr, &qt);
    }
}

template <int F>
void gemm_panels(const void* A, int lda, int M, int K, const Linear* L, int p0, int p1, const OutMap& om, int mbase,
                 int epi, const float* R, int ldr, const float* ascale = nullptr) {
    const size_t ps = (size_t)K * (F == 1 ? 32 : F == 2 ? 64 : 16);  // panel stride in W elements
    const bool sw = epi == EPI_SWIGLU;
    for (int p = p0; p < p1;) {
        const int r = p1 - p;
        int nv;
        if (sw) nv = (SHV_NV >= 4 && r >= 4) ? 4 : 2;
#ifdef SHV_GROUP_GREEDY
        else if (SHV_NV >= 4) nv = imin(4, r);
#else
        else if (SHV_NV >= 4) nv = (r == 5 || r == 6 || r == 9) ? 3 : imin(4, r);
#endif
        else nv = imin(SHV_NV, r);
        const void* W = F == 1   ? (const void*)(L->wb + (size_t)p * ps)
                        : F == 2 ? (const void*)(L->wq + (size_t)p * ps)
                                 : (const void*)(L->wp + (size_t)p * ps);
        const float* bias = L->bias ? L->bias + p * 16 : nullptr;
        const int coff = sw ? (p / 2) * 16 : p * 16;
        const float* res = R ? R + p * 16 : nullptr;
        Q8 qd{};
        const Q8* q = nullptr;
        if (F == 2) {
            qd = Q8{L->wcorr + p * 16, L->wscale + p * 16, ascale};
            q = &qd;
        }
        switch (nv) {
        case 1: gemm_rows<1, F>(A, lda, M, K, W, ps, bias, om, mbase, coff, epi, res, ldr, q); break;
        case 2: gemm_rows<2, F>(A, lda, M, K, W, ps, bias, om, mbase, coff, epi, res, ldr, q); break;
#if SHV_NV >= 3
        case 3: gemm_rows<3, F>(A, lda, M, K, W, ps, bias, om, mbase, coff, epi, res, ldr, q); break;
#endif
#if SHV_NV >= 4
        case 4: gemm_rows<4, F>(A, lda, M, K, W, ps, bias, om, mbase, coff, epi, res, ldr, q); break;
#endif
        default: break;
        }
        p += nv;
    }
}

#if SHV_LEVEL == 3
// Per-thread bf16 activation buffer (set by cnn_frame / seq_window from the thread's scratch).
__thread uint32_t* t_a16 = nullptr;
__thread size_t t_a16_cap = 0;
// rows of A (float, K even) -> bf16 pairs, round to nearest even (VCVTNE2PS2BF16)
void a_to_bf16(const float* A, int lda, int M, int K, uint32_t* out) {
    const int KP = K / 2;
    for (int m = 0; m < M; ++m) {
        const float* a = A + (size_t)m * lda;
        uint32_t* o = out + (size_t)m * KP;
        int k = 0;
        for (; k + 32 <= K; k += 32)
            _mm512_storeu_si512(o + k / 2, (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(a + k + 16), _mm512_loadu_ps(a + k)));
        for (; k + 16 <= K; k += 16)
            _mm256_storeu_si256((__m256i*)(o + k / 2), (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(a + k)));
        if (k < K) {
            const int n = K - k;  // even, < 16
            alignas(32) uint16_t tmp[16];
            _mm256_store_si256((__m256i*)tmp, (__m256i)_mm512_cvtneps_pbh(_mm512_maskz_loadu_ps((__mmask16)((1u << n) - 1), a + k)));
            memcpy(o + k / 2, tmp, sizeof(uint16_t) * n);
        }
    }
}
// rows of A (float) -> per-row symmetric INT8 (scale = max|row| / 127, 1 for an all-zero row; round to nearest
// even; clamp [-127, 127]) stored unsigned (+128) in groups of 4 (zero-padded with 128); scales -> sc
void a_to_int8(const float* A, int lda, int M, int K, uint32_t* out, float* sc) {
    const int KQ = (K + 3) / 4;
    for (int m = 0; m < M; ++m) {
        const float* a = A + (size_t)m * lda;
        __m512 mx = _mm512_setzero_ps();
        int k = 0;
        for (; k + 16 <= K; k += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(a + k)));
        if (k < K) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_maskz_loadu_ps((__mmask16)((1u << (K - k)) - 1), a + k)));
        const float amax = _mm512_reduce_max_ps(mx);
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        sc[m] = s;
        const __m512 inv = _mm512_set1_ps(1.0f / s);
        uint8_t* o = reinterpret_cast<uint8_t*>(out + (size_t)m * KQ);
        for (k = 0; k < K; k += 16) {
            const int n = imin(16, K - k);
            const __m512 x = _mm512_maskz_loadu_ps((__mmask16)((1u << n) - 1), a + k);
            __m512i v = _mm512_cvt_roundps_epi32(_mm512_mul_ps(x, inv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v = _mm512_min_epi32(_mm512_max_epi32(v, _mm512_set1_epi32(-127)), _mm512_set1_epi32(127));
            v = _mm512_add_epi32(v, _mm512_set1_epi32(128));
            alignas(16) uint8_t tmp[16];
            _mm_store_si128((__m128i*)tmp, _mm512_cvtusepi32_epi8(v));
            memcpy(o + k, tmp, n);
        }
        for (k = K; k < KQ * 4; ++k) o[k] = 128;
    }
}
__thread float* t_ascale = nullptr;  // per-row INT8 activation scales (capacity SHV_ASCALE_CAP)
#define SHV_ASCALE_CAP 4096
#endif

// Panels [p0, p1) of L (panel = 16 output columns), rows mapped by om; R (residual) is row-linear.
void gemm_map(const float* A, int lda, int M, const Linear* L, int p0, int p1, const OutMap& om, int epi,
              const float* R, int ldr) {
    if (M <= 0 || p1 <= p0) return;
#if SHV_LEVEL == 3
    if (L->fmt == FMT_BF16 && L->wb && t_a16) {
        const int KP = L->K / 2;
        const int chunk = (int)(t_a16_cap / (size_t)KP);
        for (int m0 = 0; m0 < M; m0 += chunk) {
            const int mc = imin(chunk, M - m0);
            a_to_bf16(A + (size_t)m0 * lda, lda, mc, L->K, t_a16);
            gemm_panels<1>(t_a16, KP, mc, KP, L, p0, p1, om, m0, epi, R ? R + (size_t)m0 * ldr : nullptr, ldr);
        }
        return;
    }
    if (L->fmt == FMT_INT8 && L->wq && t_a16 && t_ascale) {
        const int KQ = (L->K + 3) / 4;
        const int chunk = imin((int)(t_a16_cap / (size_t)KQ), SHV_ASCALE_CAP);
        for (int m0 = 0; m0 < M; m0 += chunk) {
            const int mc = imin(chunk, M - m0);
            a_to_int8(A + (size_t)m0 * lda, lda, mc, L->K, t_a16, t_ascale);
            gemm_panels<2>(t_a16, KQ, mc, KQ, L, p0, p1, om, m0, epi, R ? R + (size_t)m0 * ldr : nullptr, ldr,
                           t_ascale);
        }
        return;
    }
#endif
    gemm_panels<0>(A, lda, M, L->K, L, p0, p1, om, 0, epi, R, ldr);
}
// Panels [p0, p1) of L.  C/R point at column 0 of the full row-major matrices.
void gemm(const float* A, int lda, int M, const Linear* L, int p0, int p1, float* C, int ldc, int epi,
          const float* R, int ldr) {
    const OutMap om{C, ldc, 0, 0};
    gemm_map(A, lda, M, L, p0, p1, om, epi, R, ldr);
}

// Split the panels of L across a team (pairs for SwiGLU) and run the GEMM for this rank's share.
SHV_INL void gemm_team(const Team* tm, const float* A, int lda, int M, const Linear* L, float* C, int ldc, int epi,
                       const float* R = nullptr, int ldr = 0) {
    const int np = L->Np / 16;
    int a, b;
    if (epi == EPI_SWIGLU) {
        split(np / 2, tm->rank, tm->size, &a, &b);
        a *= 2; b *= 2;
    } else {
        split(np, tm->rank, tm->size, &a, &b);
    }
    gemm(A, lda, M, L, a, b, C, ldc, epi, R, ldr);
}

// ------------------------------------------------------------------------------------------------ CNN
// NHWC activations, channels padded to multiples of 16 with exact zeros.

// stem: uint8 RGB [180][320][3] -> 3x3 s2 p1 conv (BN folded) + hardswish -> [90][160][16], rows [y0,y1)
void stem_rows(const Cnn* net, const uint8_t* rgb, float* out, int y0, int y1, CnnScratch* sc) {
    const int W = SHV_W, H = SHV_H, Wo = 160;
    const int RW = (W + 2) * 3;  // padded normalised row: x' = -1 .. 320
    // 3 normalised padded input rows per output row.  (A ring that normalises each input row once was measured
    // 15 us/frame slower than this: docs/PERFORMANCE.md.)
    float* rowbuf = sc->stemrows;
    const V b = vload(net->stem_b);
    for (int y = y0; y < y1; ++y) {
        for (int dy = 0; dy < 3; ++dy) {
            float* r = rowbuf + dy * RW;
            const int iy = 2 * y - 1 + dy;
            if (iy < 0 || iy >= H) { memset(r, 0, sizeof(float) * RW); continue; }
            const uint8_t* src = rgb + (size_t)iy * W * 3;
            r[0] = r[1] = r[2] = 0.0f;
            for (int x = 0; x < W; ++x) {
                r[3 + 3 * x + 0] = net->lut[0][src[3 * x + 0]];
                r[3 + 3 * x + 1] = net->lut[1][src[3 * x + 1]];
                r[3 + 3 * x + 2] = net->lut[2][src[3 * x + 2]];
            }
            r[RW - 3] = r[RW - 2] = r[RW - 1] = 0.0f;
        }
        float* o = out + (size_t)y * Wo * 16;
        constexpr int XB = 8;
        for (int x = 0; x < Wo; x += XB) {
            V acc[XB];
#pragma GCC unroll 8
            for (int i = 0; i < XB; ++i) acc[i] = vzero();
            for (int dy = 0; dy < 3; ++dy) {
                const float* r = rowbuf + dy * RW;
                for (int dx = 0; dx < 3; ++dx) {
                    for (int c = 0; c < 3; ++c) {
                        const V w = vload(net->stem_w + ((dy * 3 + dx) * 3 + c) * 16);
#pragma GCC unroll 8
                        for (int i = 0; i < XB; ++i)
                            acc[i] = vfma(vset(r[(2 * (x + i) + dx) * 3 + c]), w, acc[i]);
                    }
                }
            }
#pragma GCC unroll 8
            for (int i = 0; i < XB; ++i) vstore(o + (x + i) * 16, vhswish(vadd(acc[i], b)));
        }
    }
}

// one depthwise output row: rows[dy] -> padded input rows (width (W+2p)*C), out [Wo][C]
template <int KS>
void dw_row(const float* const* rows, int s, int Wo, int C, const float* w, const float* b, int act, float* out) {
    constexpr int XB = 4;
    for (int c0 = 0; c0 < C; c0 += 16) {
        const V bias = vload(b + c0);
        int x = 0;
        for (; x + XB <= Wo; x += XB) {
            V acc[XB];
#pragma GCC unroll 4
            for (int i = 0; i < XB; ++i) acc[i] = vzero();
#pragma GCC unroll 5
            for (int dy = 0; dy < KS; ++dy) {
                const float* r = rows[dy] + c0;
#pragma GCC unroll 5
                for (int dx = 0; dx < KS; ++dx) {
                    const V wv = vload(w + (dy * KS + dx) * C + c0);
#pragma GCC unroll 4
                    for (int i = 0; i < XB; ++i)
                        acc[i] = vfma(vload(r + (size_t)((x + i) * s + dx) * C), wv, acc[i]);
                }
            }
#pragma GCC unroll 4
            for (int i = 0; i < XB; ++i) {
                V v = vadd(acc[i], bias);
                v = act == ACT_HS ? vhswish(v) : vmax(v, vzero());
                vstore(out + (size_t)(x + i) * C + c0, v);
            }
        }
        for (; x < Wo; ++x) {
            V acc = vzero();
            for (int dy = 0; dy < KS; ++dy) {
                const float* r = rows[dy] + c0;
                for (int dx = 0; dx < KS; ++dx)
                    acc = vfma(vload(r + (size_t)(x * s + dx) * C), vload(w + (dy * KS + dx) * C + c0), acc);
            }
            V v = vadd(acc, bias);
            v = act == ACT_HS ? vhswish(v) : vmax(v, vzero());
            vstore(out + (size_t)x * C + c0, v);
        }
    }
}

// expand (or copy) input row r of block B into its ring slot interior
SHV_INL void expand_row(const Block* B, const float* in, int r, float* slot) {
    const float* src = in + (size_t)r * B->W * B->cinp;
    float* dst = slot + (size_t)B->p * B->cexpp;
    if (!B->has_expand) {
        memcpy(dst, src, sizeof(float) * B->W * B->cinp);
        return;
    }
    gemm(src, B->cinp, B->W, &B->expand, 0, B->expand.Np / 16, dst, B->cexpp,
         B->act == ACT_HS ? EPI_HS : EPI_RELU, nullptr, 0);
}

// Streaming expand + depthwise.  Each input row is expanded exactly once into a ring of k padded rows (the
// border columns are zero); output rows must be requested in increasing order.
struct DwStream {
    const Block* B;
    const float* in;
    CnnScratch* sc;
    size_t rw;   // padded row width in floats
    int hi;      // highest input row already expanded
};
SHV_INL void dws_init(DwStream* d, const Block* B, const float* in, int y0, CnnScratch* sc) {
    d->B = B; d->in = in; d->sc = sc;
    const int p = B->p, C = B->cexpp;
    d->rw = (size_t)(B->W + 2 * p) * C;
    for (int i = 0; i < B->k; ++i) {
        float* slot = sc->ring + i * d->rw;
        memset(slot, 0, sizeof(float) * p * C);
        memset(slot + (size_t)(B->W + p) * C, 0, sizeof(float) * p * C);
    }
    d->hi = y0 * B->s - p - 1;
}
SHV_INL void dws_row(DwStream* d, int y, float* out) {
    const Block* B = d->B;
    const int k = B->k, lo = y * B->s - B->p;
    const int need = imin(lo + k - 1, B->H - 1);
    for (int r = imax(d->hi + 1, 0); r <= need; ++r) expand_row(B, d->in, r, d->sc->ring + (size_t)(r % k) * d->rw);
    if (need > d->hi) d->hi = need;
    const float* rows[5];
    for (int dy = 0; dy < k; ++dy) {
        const int r = lo + dy;
        rows[dy] = (r < 0 || r >= B->H) ? d->sc->zero : d->sc->ring + (size_t)(r % k) * d->rw;
    }
    if (k == 3) dw_row<3>(rows, B->s, B->Wo, B->cexpp, B->dw_w, B->dw_b, B->act, out);
    else dw_row<5>(rows, B->s, B->Wo, B->cexpp, B->dw_w, B->dw_b, B->act, out);
}

// Full-map expansion: the whole expanded input map in a zero-bordered team-shared buffer
// X[(H+2p)][(W+2p)][cexpp]; each rank expands its share of input rows with one GEMM, then (after a barrier)
// computes its depthwise output rows into fb->dwfull.  No input row is expanded twice.
void expand_dw_full(const Block* B, const float* in, const Team* tm, FrameBuf* fb, int y0, int y1) {
    const int p = B->p, C = B->cexpp;
    const size_t rs = (size_t)(B->W + 2 * p) * C;
    float* X = fb->xfull;
    int r0, r1;
    split(B->H, tm->rank, tm->size, &r0, &r1);
    for (int r = r0; r < r1; ++r) {
        float* row = X + (size_t)(r + p) * rs;
        memset(row, 0, sizeof(float) * p * C);
        memset(row + (size_t)(B->W + p) * C, 0, sizeof(float) * p * C);
    }
    if (tm->rank == 0) memset(X, 0, sizeof(float) * p * rs);
    if (tm->rank == tm->size - 1) memset(X + (size_t)(B->H + p) * rs, 0, sizeof(float) * p * rs);
    if (r1 > r0) {
        const float* src = in + (size_t)r0 * B->W * B->cinp;
        float* dst = X + (size_t)(r0 + p) * rs + (size_t)p * C;
        if (B->has_expand) {
            const OutMap om{dst, C, B->W, rs};
            gemm_map(src, B->cinp, (r1 - r0) * B->W, &B->expand, 0, B->expand.Np / 16, om,
                     B->act == ACT_HS ? EPI_HS : EPI_RELU, nullptr, 0);
        } else {
            for (int r = r0; r < r1; ++r)
                memcpy(dst + (size_t)(r - r0) * rs, src + (size_t)(r - r0) * B->W * B->cinp,
                       sizeof(float) * B->W * C);
        }
    }
    sync(tm);
    const float* rows[5];
    for (int y = y0; y < y1; ++y) {
        for (int dy = 0; dy < B->k; ++dy) rows[dy] = X + (size_t)(y * B->s + dy) * rs;
        float* o = fb->dwfull + (size_t)y * B->Wo * C;
        if (B->k == 3) dw_row<3>(rows, B->s, B->Wo, C, B->dw_w, B->dw_b, B->act, o);
        else dw_row<5>(rows, B->s, B->Wo, C, B->dw_w, B->dw_b, B->act, o);
    }
}

void block_run(const Block* B, const float* in, float* out, const Team* tm, FrameBuf* fb, CnnScratch* sc) {
    int y0, y1;
    split(B->Ho, tm->rank, tm->size, &y0, &y1);
    const int C = B->cexpp, Wo = B->Wo;
    const int epi = B->residual ? EPI_RES : EPI_STORE;
    const bool full = tm->size > 1 ? B->full_team : B->full_solo;
    if (full) {
        expand_dw_full(B, in, tm, fb, y0, y1);
        if (!B->has_se) {
            if (y1 > y0)
                gemm(fb->dwfull + (size_t)y0 * Wo * C, C, (y1 - y0) * Wo, &B->project, 0, B->project.Np / 16,
                     out + (size_t)y0 * Wo * B->coutp, B->coutp, epi,
                     B->residual ? in + (size_t)y0 * B->W * B->cinp : nullptr, B->cinp);
            return;
        }
    } else {
        DwStream ds;
        dws_init(&ds, B, in, y0, sc);
        if (!B->has_se) {
            for (int y = y0; y < y1; ++y) {
                dws_row(&ds, y, sc->strip);
                gemm(sc->strip, C, Wo, &B->project, 0, B->project.Np / 16, out + (size_t)y * Wo * B->coutp,
                     B->coutp, epi, B->residual ? in + (size_t)y * B->W * B->cinp : nullptr, B->cinp);
            }
            return;
        }
        // SE: phase 1 depthwise rows into the team-shared full buffer
        for (int y = y0; y < y1; ++y) dws_row(&ds, y, fb->dwfull + (size_t)y * Wo * C);
    }
    sync(tm);
    // phase 2 (every rank, identical): pooled = mean over H*W (row sums, then the sum of the row sums);
    // fc1 + ReLU; fc2 + hardsigmoid
    float* pooled = sc->vec;
    float* sq = sc->vec + 1024;
    float* scale = sc->vec + 2048;
    const int n = B->Ho * Wo;
    for (int c0 = 0; c0 < C; c0 += 16) {
        V tot = vzero();
        for (int y = 0; y < B->Ho; ++y) {
            const float* row = fb->dwfull + (size_t)y * Wo * C + c0;
            V acc = vload(row);
            for (int x = 1; x < Wo; ++x) acc = vadd(acc, vload(row + (size_t)x * C));
            tot = y == 0 ? acc : vadd(tot, acc);
        }
        vstore(pooled + c0, vdiv(tot, vset((float)n)));
    }
    gemm(pooled, C, 1, &B->se1, 0, B->se1.Np / 16, sq, B->csqp, EPI_RELU, nullptr, 0);
    gemm(sq, B->csqp, 1, &B->se2, 0, B->se2.Np / 16, scale, C, EPI_HSIG, nullptr, 0);
    sync(tm);  // every rank has read the unscaled map
    // phase 3: scale my rows in place (scale * x) and project
    for (int y = y0; y < y1; ++y) {
        float* row = fb->dwfull + (size_t)y * Wo * C;
        for (int x = 0; x < Wo; ++x)
            for (int c0 = 0; c0 < C; c0 += 16)
                vstore(row + x * C + c0, vmul(vload(scale + c0), vload(row + x * C + c0)));
    }
    if (y1 > y0)
        gemm(fb->dwfull + (size_t)y0 * Wo * C, C, (y1 - y0) * Wo, &B->project, 0, B->project.Np / 16,
             out + (size_t)y0 * Wo * B->coutp, B->coutp, epi,
             B->residual ? in + (size_t)y0 * B->W * B->cinp : nullptr, B->cinp);
}

void cnn_frame(const Cnn* net, const uint8_t* rgb, float* tokens, const Team* tm, FrameBuf* fb, CnnScratch* sc) {
#if SHV_LEVEL == 3
    t_a16 = sc->a16;
    t_a16_cap = sc->a16_cap;
    t_ascale = sc->ascale;
#endif
    int y0, y1;
    split(90, tm->rank, tm->size, &y0, &y1);
    const bool dump = net->dump && tm->rank == 0;
    PROF_BEGIN;
    stem_rows(net, rgb, fb->act[0], y0, y1, sc);
    sync(tm);
    PROF_MARK(0);
    if (dump) net->dump(net->dump_ctx, "cnn00", fb->act[0], (size_t)90 * 160 * 16);
    int cur = 0;
    for (int i = 0; i < net->nblocks; ++i) {
        const Block* B = &net->blocks[i];
        block_run(B, fb->act[cur], fb->act[cur ^ 1], tm, fb, sc);
        sync(tm);
        PROF_MARK(1 + i);
        cur ^= 1;
        if (dump) {
            char name[8] = {'c', 'n', 'n', (char)('0' + (i + 1) / 10), (char)('0' + (i + 1) % 10), 0};
            net->dump(net->dump_ctx, name, fb->act[cur], (size_t)B->Ho * B->Wo * B->coutp);
        }
    }
    // last 1x1 conv 160 -> 960 + hardswish on the 6x10 map
    const Block* L = &net->blocks[net->nblocks - 1];
    split(6, tm->rank, tm->size, &y0, &y1);
    if (y1 > y0)
        gemm(fb->act[cur] + (size_t)y0 * 10 * L->coutp, L->coutp, (y1 - y0) * 10, &net->last, 0,
             net->last.Np / 16, fb->lastmap + (size_t)y0 * 10 * 960, 960, EPI_HS, nullptr, 0);
    sync(tm);
    PROF_MARK(16);
    if (dump) net->dump(net->dump_ctx, "cnn16", fb->lastmap, (size_t)60 * 960);
    // adaptive average pool 6x10 -> 4x4 (overlapping bins), then projection 960 -> 384
    static const int r0[4] = {0, 1, 3, 4}, c0s[4] = {0, 2, 5, 7};
    int t0, t1;
    split(16, tm->rank, tm->size, &t0, &t1);
    float* pooled = sc->vec;  // up to 16 x 960
    for (int t = t0; t < t1; ++t) {
        const int ry = r0[t / 4], cx = c0s[t % 4];
        float* pr = pooled + (size_t)(t - t0) * 960;
        for (int c = 0; c < 960; c += 16) {
            V acc = vzero();
            for (int yy = ry; yy < ry + 2; ++yy)
                for (int xx = cx; xx < cx + 3; ++xx) acc = vadd(acc, vload(fb->lastmap + (size_t)(yy * 10 + xx) * 960 + c));
            vstore(pr + c, vdiv(vdiv(acc, vset(2.0f)), vset(3.0f)));
        }
    }
    if (t1 > t0)
        gemm(pooled, 960, t1 - t0, &net->proj, 0, net->proj.Np / 16, tokens + (size_t)t0 * SHV_D, SHV_D, EPI_STORE,
             nullptr, 0);
    PROF_MARK(17);
    if (net->dump) {
        sync(tm);
        if (dump) net->dump(net->dump_ctx, "tokens", tokens, (size_t)16 * SHV_D);
    }
}

// ------------------------------------------------------------------------------------------------ sequence
SHV_INL void rms_row(const float* x, float* y, const float* w, int n) {  // n multiple of 16
    V acc = vmul(vload(x), vload(x));
    for (int i = 16; i < n; i += 16) acc = vfma(vload(x + i), vload(x + i), acc);
    const float inv = 1.0f / sqrtf(vsum(acc) / (float)n + 1e-6f);
    const V vi = vset(inv);
    for (int i = 0; i < n; i += 16) vstore(y + i, vmul(vmul(vload(x + i), vi), vload(w + i)));
}
void rms_rows(const float* x, float* y, const float* w, int r0, int r1, int ld) {
    for (int r = r0; r < r1; ++r) rms_row(x + (size_t)r * ld, y + (size_t)r * ld, w, SHV_D);
}

// per-head GDN core: short conv + SiLU, L2 norm, gates, delta rule, gated RMSNorm -> O[:, h*128 ..]
void gdn_head(const Gdn* G, int h, float* P, float* O, SeqScratch* sc) {
    const int T = SHV_T;
    PROF_BEGIN;
    // causal depthwise conv (kernel 4, bias) + SiLU, in place, descending t
    const int chans[3][2] = {{GDN_Q + h * 64, 64}, {GDN_K + h * 64, 64}, {GDN_V + h * 128, 128}};
    const int wofs[3] = {h * 64, 384 + h * 64, 768 + h * 128};
    for (int g = 0; g < 3; ++g) {
        for (int c = 0; c < chans[g][1]; c += 16) {
            const int col = chans[g][0] + c, wc = wofs[g] + c;
            const V w0 = vload(G->conv_w + 0 * 1536 + wc), w1 = vload(G->conv_w + 1 * 1536 + wc);
            const V w2 = vload(G->conv_w + 2 * 1536 + wc), w3 = vload(G->conv_w + 3 * 1536 + wc);
            const V b = vload(G->conv_b + wc);
            for (int t = T - 1; t >= 0; --t) {
                V acc = vzero();
                if (t >= 3) acc = vfma(vload(P + (size_t)(t - 3) * GDN_P + col), w0, acc);
                if (t >= 2) acc = vfma(vload(P + (size_t)(t - 2) * GDN_P + col), w1, acc);
                if (t >= 1) acc = vfma(vload(P + (size_t)(t - 1) * GDN_P + col), w2, acc);
                acc = vfma(vload(P + (size_t)t * GDN_P + col), w3, acc);
                vstore(P + (size_t)t * GDN_P + col, vsilu(vadd(acc, b)));
            }
        }
    }
    PROF_MARK(50);
    // L2 norm of q and k: x * (1 / sqrt(sum x^2 + 1e-6)); q additionally scaled by 64^-0.5 = 0.125 (exact);
    // gates: decay = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
    float* decay = sc->rows;        // [321]
    float* beta = sc->rows + 384;   // [321]
    for (int t = 0; t < T; ++t) {
        float* row = P + (size_t)t * GDN_P;
        float* q = row + GDN_Q + h * 64;
        float* k = row + GDN_K + h * 64;
        V aq = vmul(vload(q), vload(q)), ak = vmul(vload(k), vload(k));
        for (int i = 16; i < 64; i += 16) {
            aq = vfma(vload(q + i), vload(q + i), aq);
            ak = vfma(vload(k + i), vload(k + i), ak);
        }
        const V iq = vset(1.0f / sqrtf(vsum(aq) + 1e-6f)), ik = vset(1.0f / sqrtf(vsum(ak) + 1e-6f));
        for (int i = 0; i < 64; i += 16) {
            vstore(q + i, vmul(vmul(vload(q + i), iq), vset(0.125f)));
            vstore(k + i, vmul(vload(k + i), ik));
        }
        decay[t] = expf(G->neg_rate[h] * softplus_f(row[GDN_A + h] + G->dt_bias[h]));
        beta[t] = sigmoid_f(row[GDN_B + h]);
    }
    PROF_MARK(52);
    // Recurrence, one pass over the state per token:
    //   u_t = beta_t (v_t - a_t (S_{t-1}^T k_t));  S_t = a_t S_{t-1} + k_t u_t^T;  o_t = S_t^T q_t
    // and in the same pass p = S_t^T k_{t+1} for the next token.  Zero initial state.
    float* st = sc->state;  // [64][128]
    memset(st, 0, sizeof(float) * 64 * 128);
    V pred[8];
#pragma GCC unroll 8
    for (int z = 0; z < 8; ++z) pred[z] = vzero();
    for (int t = 0; t < T; ++t) {
        const float* row = P + (size_t)t * GDN_P;
        const float* q = row + GDN_Q + h * 64;
        const float* k = row + GDN_K + h * 64;
        const float* v = row + GDN_V + h * 128;
        const float* kn = t + 1 < T ? row + GDN_P + GDN_K + h * 64 : k;
        const V a = vset(decay[t]), b = vset(beta[t]);
        V u[8], o[8];
#pragma GCC unroll 8
        for (int z = 0; z < 8; ++z) {
            u[z] = vmul(vsub(vload(v + z * 16), vmul(a, pred[z])), b);
            o[z] = vzero();
            pred[z] = vzero();
        }
        for (int j = 0; j < 64; ++j) {
            const V kj = vset(k[j]), qj = vset(q[j]), knj = vset(kn[j]);
            float* s = st + j * 128;
#pragma GCC unroll 8
            for (int z = 0; z < 8; ++z) {
                const V x = vfma(kj, u[z], vmul(vload(s + z * 16), a));
                vstore(s + z * 16, x);
                o[z] = vfma(x, qj, o[z]);
                pred[z] = vfma(x, knj, pred[z]);
            }
        }
        float* out = O + (size_t)t * SHV_VD + h * 128;
#pragma GCC unroll 8
        for (int z = 0; z < 8; ++z) vstore(out + z * 16, o[z]);
    }
    PROF_MARK(53);
    // gated RMSNorm: ((o * rstd) * w) * silu(gate)
    const float* ow = G->o_norm_w;
    for (int t = 0; t < T; ++t) {
        float* out = O + (size_t)t * SHV_VD + h * 128;
        const float* gate = P + (size_t)t * GDN_P + GDN_G + h * 128;
        V acc = vmul(vload(out), vload(out));
        for (int z = 1; z < 8; ++z) acc = vfma(vload(out + z * 16), vload(out + z * 16), acc);
        const V rstd = vset(1.0f / sqrtf(vsum(acc) / 128.0f + 1e-6f));
        for (int z = 0; z < 8; ++z)
            vstore(out + z * 16,
                   vmul(vmul(vmul(vload(out + z * 16), rstd), vload(ow + z * 16)), vsilu(vload(gate + z * 16))));
    }
    PROF_MARK(51);
}

void gdn_stage(const Gdn* G, const Team* tm, SeqBuf* B, SeqScratch* sc, int pid) {
    (void)pid;  // profile stage id base (profile builds)
    int r0, r1;
    split(SHV_T, tm->rank, tm->size, &r0, &r1);
    PROF_BEGIN;
    for (int rep = 0; rep < G->repeats; ++rep) {
        rms_rows(B->h, B->hn, G->norm_w, r0, r1, SHV_D);
        sync(tm);
        PROF_MARK(pid + 0);
        gemm_team(tm, B->hn, SHV_D, SHV_T, &G->in, B->P, GDN_P, EPI_STORE);
        sync(tm);
        PROF_MARK(pid + 1);
        for (int h = tm->rank; h < SHV_HEADS; h += tm->size) gdn_head(G, h, B->P, B->O, sc);
        sync(tm);
        PROF_MARK(pid + 2);
        gemm_team(tm, B->O, SHV_VD, SHV_T, &G->out, B->h, SHV_D, G->repeats == 2 ? EPI_ACC_DIV2 : EPI_ACC);
        sync(tm);
        PROF_MARK(pid + 3);
    }
    rms_rows(B->h, B->hn, G->ffn_norm_w, r0, r1, SHV_D);
    sync(tm);
    PROF_MARK(pid + 0);
    gemm_team(tm, B->hn, SHV_D, SHV_T, &G->gu, B->F, SHV_FF, EPI_SWIGLU);
    sync(tm);
    PROF_MARK(pid + 4);
    gemm_team(tm, B->F, SHV_FF, SHV_T, &G->down, B->h, SHV_D, EPI_ACC);
    sync(tm);
    PROF_MARK(pid + 5);
}

// SISA head h for the last query row (t = 320): cumulative gates (double accumulation, rounded once per prefix,
// as torch.cumsum on CPU), pair rotations, scores, softmax, y_h = sum_t p_t v_t -> y[h*64 ..]
void sisa_head(const Sisa* S, int h, float* P, const float* q320, float* c320, float* y, SeqScratch* sc) {
    const int T = SHV_T;
    float* gt = sc->rows;          // [321]
    float* ft = sc->rows + 384;    // [321]
    float* sco = sc->rows + 768;   // [321]
    double g = 0.0, f = 0.0, phi[16];
    for (int i = 0; i < 16; ++i) phi[i] = 0.0;
    for (int t = 0; t < T; ++t) {
        float* row = P + (size_t)t * SISA_P;
        g += (double)(-softplus_f(row[SISA_AL + h]));
        f += (double)logsigmoid_f(row[SISA_FG + h]);
        gt[t] = (float)g;
        ft[t] = (float)f;
        float* b = row + SISA_B + h * 32;
        const float* th = row + SISA_TH + h * 16;
        for (int pr = 0; pr < 16; ++pr) {
            phi[pr] += (double)th[pr];
            const float ph = (float)phi[pr];
            float sn, cs;
            sincosf(ph, &sn, &cs);
            const float e = b[2 * pr], o = b[2 * pr + 1];
            b[2 * pr] = e * cs - o * sn;
            b[2 * pr + 1] = e * sn + o * cs;
            if (t == T - 1) {
                float* c = c320 + h * 32;
                const float ce = c[2 * pr], co = c[2 * pr + 1];
                c[2 * pr] = ce * cs - co * sn;
                c[2 * pr + 1] = ce * sn + co * cs;
            }
        }
    }
    const float* q = q320 + h * 64;
    const float* c = c320 + h * 32;
    const float lam = S->lam[h];
    const float gl = gt[T - 1], fl = ft[T - 1];
    float mx = -INFINITY;
    const V q0 = vload(q), q1 = vload(q + 16), q2 = vload(q + 32), q3 = vload(q + 48);
    const V c0 = vload(c), c1 = vload(c + 16);
    for (int t = 0; t < T; ++t) {
        const float* row = P + (size_t)t * SISA_P;
        const float* kk = row + SISA_K + h * 64;
        const float* bb = row + SISA_B + h * 32;
        V a = vmul(q0, vload(kk));
        a = vfma(q1, vload(kk + 16), a);
        a = vfma(q2, vload(kk + 32), a);
        a = vfma(q3, vload(kk + 48), a);
        V s = vmul(c0, vload(bb));
        s = vfma(c1, vload(bb + 16), s);
        float score = vsum(a) * 0.125f;
        score = score + (lam * expf(gl - gt[t])) * vsum(s);
        score = score + (fl - ft[t]);
        sco[t] = score;
        if (score > mx) mx = score;
    }
    double sum = 0.0;
    for (int t = 0; t < T; ++t) { sco[t] = expf(sco[t] - mx); sum += (double)sco[t]; }
    const float inv = 1.0f / (float)sum;
    // y = sum_t p_t v_t, in blocks of SHV_KC rows (block sums added in order)
    V y0 = vzero(), y1 = vzero(), y2 = vzero(), y3 = vzero();
    for (int t0 = 0; t0 < T; t0 += SHV_KC) {
        V b0 = vzero(), b1 = vzero(), b2 = vzero(), b3 = vzero();
        for (int t = t0; t < imin(T, t0 + SHV_KC); ++t) {
            const V p = vset(sco[t] * inv);
            const float* vv = P + (size_t)t * SISA_P + SISA_V + h * 64;
            b0 = vfma(p, vload(vv), b0);
            b1 = vfma(p, vload(vv + 16), b1);
            b2 = vfma(p, vload(vv + 32), b2);
            b3 = vfma(p, vload(vv + 48), b3);
        }
        if (t0 == 0) { y0 = b0; y1 = b1; y2 = b2; y3 = b3; }
        else { y0 = vadd(y0, b0); y1 = vadd(y1, b1); y2 = vadd(y2, b2); y3 = vadd(y3, b3); }
    }
    vstore(y + h * 64, y0); vstore(y + h * 64 + 16, y1); vstore(y + h * 64 + 32, y2); vstore(y + h * 64 + 48, y3);
}

void seq_window(const Seq* S, const float* const* ft, float* logit, const Team* tm, SeqBuf* B, SeqScratch* sc) {
#if SHV_LEVEL == 3
    t_a16 = sc->a16;
    t_a16_cap = sc->a16_cap;
#endif
    int r0, r1;
    split(SHV_T, tm->rank, tm->size, &r0, &r1);
    for (int i = r0; i < r1; ++i) {
        float* h = B->h + (size_t)i * SHV_D;
        if (i == SHV_T - 1) { memcpy(h, S->readout, sizeof(float) * SHV_D); continue; }
        const float* tok = ft[i / 16] + (size_t)(i % 16) * SHV_D;
        const float* pos = S->pos + (size_t)i * SHV_D;
        for (int d = 0; d < SHV_D; d += 16) vstore(h + d, vadd(vload(tok + d), vload(pos + d)));
    }
    sync(tm);
    const bool dump = S->dump && tm->rank == 0;
    if (dump) S->dump(S->dump_ctx, "h0", B->h, (size_t)SHV_T * SHV_D);
    gdn_stage(&S->gdn[0], tm, B, sc, 20);
    if (dump) S->dump(S->dump_ctx, "after_gdn_a", B->h, (size_t)SHV_T * SHV_D);
    gdn_stage(&S->gdn[1], tm, B, sc, 30);
    if (dump) S->dump(S->dump_ctx, "after_gdn_b", B->h, (size_t)SHV_T * SHV_D);
    // SISA block, last row only
    const Sisa* A = &S->sisa;
    PROF_BEGIN;
    rms_rows(B->h, B->hn, A->attn_norm_w, r0, r1, SHV_D);
    sync(tm);
    gemm_team(tm, B->hn, SHV_D, SHV_T, &A->in, B->P, SISA_P, EPI_STORE);
    PROF_MARK(40);
    float* q320 = B->vec;            // 384
    float* c320 = B->vec + 384;      // 192
    float* y = B->vec + 768;         // 384
    float* x = B->vec + 1152;        // 384
    float* xn = B->vec + 1536;       // 384
    float* m = B->vec + 1920;        // 960
    float* hid = B->vec + 2880;      // 128
    const float* last = B->hn + (size_t)(SHV_T - 1) * SHV_D;
    if (tm->rank == (tm->size > 1 ? 1 : 0)) {
        gemm(last, SHV_D, 1, &A->q, 0, A->q.Np / 16, q320, SHV_D, EPI_STORE, nullptr, 0);
        gemm(last, SHV_D, 1, &A->c, 0, A->c.Np / 16, c320, 192, EPI_STORE, nullptr, 0);
    }
    sync(tm);
    PROF_MARK(41);
    for (int h = tm->rank; h < SHV_HEADS; h += tm->size) sisa_head(A, h, B->P, q320, c320, y, sc);
    sync(tm);
    PROF_MARK(42);
    if (tm->rank == 0) {
        memcpy(x, B->h + (size_t)(SHV_T - 1) * SHV_D, sizeof(float) * SHV_D);
        gemm(y, SHV_D, 1, &A->o, 0, A->o.Np / 16, x, SHV_D, EPI_ACC, nullptr, 0);
    }
    sync(tm);
    if (tm->rank == 0) rms_row(x, xn, A->ffn_norm_w, SHV_D);
    sync(tm);
    gemm_team(tm, xn, SHV_D, 1, &A->gu, m, SHV_FF, EPI_SWIGLU);
    sync(tm);
    gemm_team(tm, m, SHV_FF, 1, &A->down, x, SHV_D, EPI_ACC);
    sync(tm);
    if (tm->rank == 0) {
        if (dump) S->dump(S->dump_ctx, "after_sisa_last", x, SHV_D);
        rms_row(x, xn, S->final_norm_w, SHV_D);
        gemm(xn, SHV_D, 1, &S->head0, 0, S->head0.Np / 16, hid, 128, EPI_SILU, nullptr, 0);
        V acc = vmul(vload(hid), vload(S->head2_w));
        for (int i = 16; i < 128; i += 16) acc = vfma(vload(hid + i), vload(S->head2_w + i), acc);
        *logit = vsum(acc) + S->head2_b;
        if (dump) {
            S->dump(S->dump_ctx, "final_norm_row", xn, SHV_D);
            S->dump(S->dump_ctx, "head_hidden", hid, 128);
            S->dump(S->dump_ctx, "logit", logit, 1);
        }
    }
    PROF_MARK(43);
}

const IsaOps ops = {SHV_NAME, cnn_frame, seq_window, gemm, prof_get};

}  // namespace
