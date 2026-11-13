// SHIELD-V CUDA engine: GEMM kernels  C[m,n] = epilogue( sum_k A[m,k] * W[n,k] ).
//
// A: activations, row-major [M,K] (lda).  W: weights, row-major [N,K] (ldb) = PyTorch Linear / 1x1-conv layout.
// Two backends with one parameter block:
//   * gemm_simt<float>: FFMA, IEEE fp32 (no TF32).  Every output element is accumulated by one thread with
//     acc = fma(a_k, w_k, acc) for k = 0..K-1 in order.
//   * gemm_mma<bf16|fp16>: mma.sync m16n8k16 tensor cores, fp32 accumulators, k processed in 16-wide chunks
//     starting at k=0 in order (zero-filled tail).
// Neither backend splits K, and the k order does not depend on the tile configuration, so an output row is
// bitwise independent of M, of the tile shape chosen for M, and of the other rows in the batch.
#pragma once
#include "common.cuh"

namespace shv {

struct GemmParams {
    int M = 0, N = 0, K = 0;
    const void* A = nullptr; long long lda = 0;
    const void* B = nullptr; long long ldb = 0;
    // optional SE prologue: A[m,k] *= a_scale[(m / scale_rows) * ld_scale + k]   (fp32 scale)
    const float* a_scale = nullptr; int scale_rows = 1; int ld_scale = 0;
    const float* bias = nullptr;  // [N]
    int act = ACT_NONE;
    void* C = nullptr; long long ldc = 0; int c_float = 0;  // C is float (else storage type T)
    // optional residual: out = R + v (r_div == 1) or R + v / r_div; R may alias C (read-before-write per element)
    const void* R = nullptr; long long ldr = 0; int r_float = 0; float r_div = 1.0f;
    int swiglu = 0;  // columns (2j, 2j+1) = (gate_j, up_j): out[m, j] = silu(gate) * up, written at column j
};

template <typename T>
__device__ __forceinline__ void epi_pair(const GemmParams& p, int m, int n, float v0, float v1) {
    if (m >= p.M || n >= p.N) return;
    if (p.swiglu) {
        const float out = silu(v0) * v1;
        const long long idx = (long long)m * p.ldc + (n >> 1);
        if (p.c_float) static_cast<float*>(p.C)[idx] = out;
        else static_cast<T*>(p.C)[idx] = from_f<T>(out);
        return;
    }
    const bool has1 = n + 1 < p.N;
    if (p.bias) { v0 += p.bias[n]; if (has1) v1 += p.bias[n + 1]; }
    v0 = apply_act(v0, p.act);
    v1 = apply_act(v1, p.act);
    if (p.R) {
        const long long r = (long long)m * p.ldr + n;
        float r0, r1 = 0.0f;
        if (p.r_float) { r0 = static_cast<const float*>(p.R)[r]; if (has1) r1 = static_cast<const float*>(p.R)[r + 1]; }
        else { r0 = to_f(static_cast<const T*>(p.R)[r]); if (has1) r1 = to_f(static_cast<const T*>(p.R)[r + 1]); }
        if (p.r_div != 1.0f) { v0 = r0 + v0 / p.r_div; v1 = r1 + v1 / p.r_div; }
        else { v0 = r0 + v0; v1 = r1 + v1; }
    }
    const long long idx = (long long)m * p.ldc + n;
    if (p.c_float) {
        float* C = static_cast<float*>(p.C);
        if (has1 && !(idx & 1)) *reinterpret_cast<float2*>(C + idx) = make_float2(v0, v1);
        else { C[idx] = v0; if (has1) C[idx + 1] = v1; }
    } else {
        T* C = static_cast<T*>(p.C);
        C[idx] = from_f<T>(v0);
        if (has1) C[idx + 1] = from_f<T>(v1);
    }
}

// ================================================================================================ SIMT fp32
// Tile BM x BN x 8, thread micro-tile TM x TN made of 4x4 groups strided over the tile (conflict-free float4 reads).
template <int BM, int BN, int TM, int TN, bool SCALE>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_simt(GemmParams p) {
    constexpr int BK = 8;
    constexpr int NTHREADS = (BM / TM) * (BN / TN);
    constexpr int TX = BN / TN;
    constexpr int RG = TM / 4, CG = TN / 4;
    constexpr int RS = BM / RG, CS = BN / CG;
    constexpr int A_F4 = BM * BK / 4, B_F4 = BN * BK / 4;
    constexpr int A_PER = (A_F4 + NTHREADS - 1) / NTHREADS, B_PER = (B_F4 + NTHREADS - 1) / NTHREADS;
    __shared__ __align__(16) float As[2][BK][BM + 4];
    __shared__ __align__(16) float Bs[2][BK][BN + 4];
    const int tid = threadIdx.x, tx = tid % TX, ty = tid / TX;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const float* A = static_cast<const float*>(p.A);
    const float* B = static_cast<const float*>(p.B);
    float4 ra[A_PER], rb[B_PER];

    auto load = [&](int k0) {
#pragma unroll
        for (int i = 0; i < A_PER; ++i) {
            const int id = tid + i * NTHREADS;
            ra[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (id < A_F4) {
                const int row = id >> 1, kc = (id & 1) * 4, gm = m0 + row, gk = k0 + kc;
                if (gm < p.M && gk < p.K) {
                    ra[i] = __ldg(reinterpret_cast<const float4*>(A + (long long)gm * p.lda + gk));
                    if (SCALE) {
                        const float* s = p.a_scale + (long long)(gm / p.scale_rows) * p.ld_scale + gk;
                        ra[i].x *= s[0]; ra[i].y *= s[1]; ra[i].z *= s[2]; ra[i].w *= s[3];
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < B_PER; ++i) {
            const int id = tid + i * NTHREADS;
            rb[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (id < B_F4) {
                const int row = id >> 1, kc = (id & 1) * 4, gn = n0 + row, gk = k0 + kc;
                if (gn < p.N && gk < p.K) rb[i] = __ldg(reinterpret_cast<const float4*>(B + (long long)gn * p.ldb + gk));
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < A_PER; ++i) {
            const int id = tid + i * NTHREADS;
            if (id < A_F4) {
                const int row = id >> 1, kc = (id & 1) * 4;
                As[buf][kc + 0][row] = ra[i].x; As[buf][kc + 1][row] = ra[i].y;
                As[buf][kc + 2][row] = ra[i].z; As[buf][kc + 3][row] = ra[i].w;
            }
        }
#pragma unroll
        for (int i = 0; i < B_PER; ++i) {
            const int id = tid + i * NTHREADS;
            if (id < B_F4) {
                const int row = id >> 1, kc = (id & 1) * 4;
                Bs[buf][kc + 0][row] = rb[i].x; Bs[buf][kc + 1][row] = rb[i].y;
                Bs[buf][kc + 2][row] = rb[i].z; Bs[buf][kc + 3][row] = rb[i].w;
            }
        }
    };

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int KT = (p.K + BK - 1) / BK;
    load(0);
    store(0);
    __syncthreads();
    for (int kt = 0; kt < KT; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < KT) load((kt + 1) * BK);
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
#pragma unroll
            for (int g = 0; g < RG; ++g) {
                const float4 v = *reinterpret_cast<const float4*>(&As[buf][kk][g * RS + ty * 4]);
                a[g * 4 + 0] = v.x; a[g * 4 + 1] = v.y; a[g * 4 + 2] = v.z; a[g * 4 + 3] = v.w;
            }
#pragma unroll
            for (int g = 0; g < CG; ++g) {
                const float4 v = *reinterpret_cast<const float4*>(&Bs[buf][kk][g * CS + tx * 4]);
                b[g * 4 + 0] = v.x; b[g * 4 + 1] = v.y; b[g * 4 + 2] = v.z; b[g * 4 + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (kt + 1 < KT) {
            store(buf ^ 1);
            __syncthreads();
        }
    }
    // Compact epilogue: TM passes; pass i stages micro-tile row i of every thread row (BM/TM rows x BN) in smem,
    // then a non-unrolled loop writes coalesced pairs (one copy of the epilogue code instead of TM*TN/2).
    constexpr int TY = BM / TM;
    static_assert(TY * BN <= 2 * BK * (BM + 4), "epilogue slab");
    __syncthreads();
    float* slab = &As[0][0][0];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int g = 0; g < CG; ++g)
            *reinterpret_cast<float4*>(slab + ty * BN + g * CS + tx * 4) =
                make_float4(acc[i][g * 4], acc[i][g * 4 + 1], acc[i][g * 4 + 2], acc[i][g * 4 + 3]);
        __syncthreads();
#pragma unroll 1
        for (int q = tid; q < TY * BN / 2; q += NTHREADS) {
            const int r = q / (BN / 2), c = (q % (BN / 2)) * 2;
            const float2 v = *reinterpret_cast<const float2*>(slab + r * BN + c);
            epi_pair<float>(p, m0 + (i / 4) * RS + r * 4 + (i % 4), n0 + c, v.x, v.y);
        }
        __syncthreads();
    }
}


// Small-M fp32 path: one thread per output pair, acc = fmaf(A[m,k], W[n,k], acc) for k = 0..K-1 in order and the
// same prologue scaling and epilogue as gemm_simt, so its results are bitwise identical to gemm_simt's.
template <bool SCALE>
__global__ void __launch_bounds__(128) gemv_seq(GemmParams p) {
    const long long idx = blockIdx.x * 128LL + threadIdx.x;
    const int np = (p.N + 1) / 2;
    if (idx >= (long long)p.M * np) return;
    const int m = int(idx / np), n = int(idx % np) * 2;
    const float* a = static_cast<const float*>(p.A) + (long long)m * p.lda;
    const float* w0 = static_cast<const float*>(p.B) + (long long)n * p.ldb;
    const float* w1 = n + 1 < p.N ? w0 + p.ldb : w0;
    const float* sc = SCALE ? p.a_scale + (long long)(m / p.scale_rows) * p.ld_scale : nullptr;
    float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll 4
    for (int k = 0; k < p.K; k += 4) {
        float4 av = __ldg(reinterpret_cast<const float4*>(a + k));
        if (SCALE) { av.x *= sc[k]; av.y *= sc[k + 1]; av.z *= sc[k + 2]; av.w *= sc[k + 3]; }
        const float4 b0 = __ldg(reinterpret_cast<const float4*>(w0 + k)), b1 = __ldg(reinterpret_cast<const float4*>(w1 + k));
        acc0 = fmaf(av.x, b0.x, acc0); acc1 = fmaf(av.x, b1.x, acc1);
        acc0 = fmaf(av.y, b0.y, acc0); acc1 = fmaf(av.y, b1.y, acc1);
        acc0 = fmaf(av.z, b0.z, acc0); acc1 = fmaf(av.z, b1.z, acc1);
        acc0 = fmaf(av.w, b0.w, acc0); acc1 = fmaf(av.w, b1.w, acc1);
    }
    epi_pair<float>(p, m, n, acc0, acc1);
}

// ================================================================================================ tensor cores
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t* r, const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

// Pure register operation: not volatile, so the compiler may interleave it with the next fragment loads.
template <typename T> __device__ __forceinline__ void mma16816(float* d, const uint32_t* a, const uint32_t* b);
template <> __device__ __forceinline__ void mma16816<__nv_bfloat16>(float* d, const uint32_t* a, const uint32_t* b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <> __device__ __forceinline__ void mma16816<__half>(float* d, const uint32_t* a, const uint32_t* b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <typename T> struct Pair2;
template <> struct Pair2<__nv_bfloat16> {
    using type = __nv_bfloat162;
    static __device__ __forceinline__ type make(float a, float b) { return __floats2bfloat162_rn(a, b); }
    static __device__ __forceinline__ float2 get(type v) { return __bfloat1622float2(v); }
};
template <> struct Pair2<__half> {
    using type = __half2;
    static __device__ __forceinline__ type make(float a, float b) { return __floats2half2_rn(a, b); }
    static __device__ __forceinline__ float2 get(type v) { return __half22float2(v); }
};

constexpr int MMA_BK = 32, MMA_PITCH = MMA_BK + 8;  // 80-byte smem rows: conflict-free ldmatrix

template <int BM, int BN, int STAGES>
constexpr int mma_smem_bytes() { return STAGES * (BM + BN) * MMA_PITCH * 2; }  // >= epilogue staging (checked below)

// Final value of output element pair (m, n), (m, n+1) after bias, activation and residual (n even, n < N).
template <typename T>
__device__ __forceinline__ void epi_values(const GemmParams& p, int m, int n, float& v0, float& v1) {
    const bool has1 = n + 1 < p.N;
    if (p.bias) { v0 += p.bias[n]; if (has1) v1 += p.bias[n + 1]; }
    v0 = apply_act(v0, p.act);
    v1 = apply_act(v1, p.act);
    if (p.R) {
        const long long r = (long long)m * p.ldr + n;
        float r0, r1 = 0.0f;
        if (p.r_float) {
            r0 = static_cast<const float*>(p.R)[r];
            if (has1) r1 = static_cast<const float*>(p.R)[r + 1];
        } else if (has1 && !(r & 1)) {
            const float2 f = Pair2<T>::get(*reinterpret_cast<const typename Pair2<T>::type*>(static_cast<const T*>(p.R) + r));
            r0 = f.x; r1 = f.y;
        } else {
            r0 = to_f(static_cast<const T*>(p.R)[r]);
            if (has1) r1 = to_f(static_cast<const T*>(p.R)[r + 1]);
        }
        if (p.r_div != 1.0f) { v0 = r0 + v0 / p.r_div; v1 = r1 + v1 / p.r_div; }
        else { v0 = r0 + v0; v1 = r1 + v1; }
    }
}

// Block tile BM x BN x 32, warp tile WM x WN, STAGES-deep cp.async pipeline, register double-buffered fragments.
// Storage-type outputs are staged through shared memory and written with 16-byte stores; fp32 outputs use float2.
template <typename T, int BM, int BN, int WM, int WN, int STAGES, bool SCALE>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
gemm_mma(GemmParams p) {
    constexpr int WARPS_N = BN / WN;
    constexpr int NTHREADS = (BM / WM) * (BN / WN) * 32;
    constexpr int MT = WM / 16, NT = WN / 8;
    static_assert(NT % 2 == 0 || NT == 1, "NT");
    static_assert((BM / WM) * (BN / WN) * 16 * (WN + 4) * 4 <= STAGES * (BM + BN) * MMA_PITCH * 2, "epilogue staging");
    static_assert(WN <= 64 && 32 % (WN / 2) == 0, "epilogue lane mapping");
    constexpr int A_CHUNKS = BM * MMA_BK / 8, B_CHUNKS = BN * MMA_BK / 8;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* As = reinterpret_cast<T*>(smem_raw);
    T* Bs = As + STAGES * BM * MMA_PITCH;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm0 = (warp / WARPS_N) * WM, wn0 = (warp % WARPS_N) * WN;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const T* A = static_cast<const T*>(p.A);
    const T* B = static_cast<const T*>(p.B);
    const int KT = (p.K + MMA_BK - 1) / MMA_BK;

    auto load_stage = [&](int stage, int kt) {
        const int k0 = kt * MMA_BK;
        T* as = As + stage * BM * MMA_PITCH;
        T* bs = Bs + stage * BN * MMA_PITCH;
#pragma unroll
        for (int c = tid; c < A_CHUNKS; c += NTHREADS) {
            const int row = c >> 2, kc = (c & 3) * 8, gm = m0 + row, gk = k0 + kc;
            const bool ok = gm < p.M && gk < p.K;
            const T* src = ok ? A + (long long)gm * p.lda + gk : A;
            if (SCALE) {
                uint4 u = make_uint4(0, 0, 0, 0);
                if (ok) {
                    u = __ldg(reinterpret_cast<const uint4*>(src));
                    T* h = reinterpret_cast<T*>(&u);
                    const float* s = p.a_scale + (long long)(gm / p.scale_rows) * p.ld_scale + gk;
#pragma unroll
                    for (int e = 0; e < 8; ++e) h[e] = from_f<T>(to_f(h[e]) * s[e]);
                }
                *reinterpret_cast<uint4*>(as + row * MMA_PITCH + kc) = u;
            } else {
                cp_async16(as + row * MMA_PITCH + kc, src, ok);
            }
        }
#pragma unroll
        for (int c = tid; c < B_CHUNKS; c += NTHREADS) {
            const int row = c >> 2, kc = (c & 3) * 8, gn = n0 + row, gk = k0 + kc;
            const bool ok = gn < p.N && gk < p.K;
            cp_async16(bs + row * MMA_PITCH + kc, ok ? B + (long long)gn * p.ldb + gk : B, ok);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    auto load_frags = [&](const T* as, const T* bs, int ks, uint32_t (*af)[4], uint32_t (*bf)[2]) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            ldmatrix_x4(af[mt], as + (wm0 + mt * 16 + (lane & 15)) * MMA_PITCH + ks * 16 + (lane >> 4) * 8);
        if (NT == 1) {
            ldmatrix_x2(bf[0], bs + (wn0 + (lane & 7)) * MMA_PITCH + ks * 16 + ((lane >> 3) & 1) * 8);
        } else {
#pragma unroll
            for (int nt = 0; nt < NT; nt += 2) {
                uint32_t r[4];
                ldmatrix_x4(r, bs + (wn0 + nt * 8 + (lane & 7) + ((lane >> 4) << 3)) * MMA_PITCH + ks * 16 + ((lane >> 3) & 1) * 8);
                bf[nt][0] = r[0]; bf[nt][1] = r[1];
                if (nt + 1 < NT) { bf[nt + 1][0] = r[2]; bf[nt + 1][1] = r[3]; }
            }
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_stage(s, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int nk = kt + STAGES - 1;
        if (nk < KT) load_stage(nk % STAGES, nk);
        cp_async_commit();
        const T* as = As + (kt % STAGES) * BM * MMA_PITCH;
        const T* bs = Bs + (kt % STAGES) * BN * MMA_PITCH;
        uint32_t af[2][MT][4], bf[2][NT][2];
        load_frags(as, bs, 0, af[0], bf[0]);
#pragma unroll
        for (int ks = 0; ks < MMA_BK / 16; ++ks) {
            if (ks + 1 < MMA_BK / 16) load_frags(as, bs, ks + 1, af[(ks + 1) & 1], bf[(ks + 1) & 1]);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) mma16816<T>(acc[mt][nt], af[ks & 1][mt], bf[ks & 1][nt]);
        }
    }
    cp_async_wait<0>();
    __syncthreads();  // all warps are done reading the pipeline buffers; reuse them for the epilogue
    // Compact epilogue: each warp stages 16 accumulator rows at a time in its own smem region, then a
    // non-unrolled loop applies bias / activation / residual / swiglu and writes coalesced pairs.
    constexpr int SP = WN + 4;            // staging pitch (floats)
    constexpr int PAIRS = WN / 2;         // column pairs per row
    constexpr int RPP = 32 / PAIRS;       // rows per pass of 32 lanes
    float* st = reinterpret_cast<float*>(smem_raw) + warp * 16 * SP;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int prow = lane / PAIRS, pcol = (lane % PAIRS) * 2;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            *reinterpret_cast<float2*>(st + g * SP + nt * 8 + c2) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
            *reinterpret_cast<float2*>(st + (g + 8) * SP + nt * 8 + c2) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
        }
        __syncwarp();
        if (lane < RPP * PAIRS) {
#pragma unroll 1
            for (int r = prow; r < 16; r += RPP) {
                const int m = m0 + wm0 + mt * 16 + r, n = n0 + wn0 + pcol;
                if (m < p.M && n < p.N) {
                    const float2 v = *reinterpret_cast<const float2*>(st + r * SP + pcol);
                    if (p.c_float) {
                        epi_pair<T>(p, m, n, v.x, v.y);
                    } else if (p.swiglu) {
                        static_cast<T*>(p.C)[(long long)m * p.ldc + (n >> 1)] = from_f<T>(silu(v.x) * v.y);
                    } else {
                        float v0 = v.x, v1 = v.y;
                        epi_values<T>(p, m, n, v0, v1);
                        const long long idx = (long long)m * p.ldc + n;
                        T* C = static_cast<T*>(p.C);
                        if (n + 1 < p.N && !(idx & 1)) *reinterpret_cast<typename Pair2<T>::type*>(C + idx) = Pair2<T>::make(v0, v1);
                        else { C[idx] = from_f<T>(v0); if (n + 1 < p.N) C[idx + 1] = from_f<T>(v1); }
                    }
                }
            }
        }
        __syncwarp();
    }
}

// Host dispatcher (gemm.cu). Chooses a tile shape from (M, N, K); never changes the k order (see top).
template <typename T> void gemm(const GemmParams& p, cudaStream_t stream);
void gemm_init_attributes();
// Test only: -1 automatic; fp32: 1 gemv_seq, 2 simt 64x64; half: 3 mma 32x64, 4 mma 128x128.
void gemm_force_kernel(int k);

}  // namespace shv
