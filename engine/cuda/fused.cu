// SHIELD-V CUDA engine: fused MobileNetV3 expand (1x1 conv + folded BN + act) -> depthwise (KxK + BN + act) for the
// tensor-core modes.  The expanded tensor never leaves shared memory.
//
// Bitwise identical to the unfused path (gemm_mma expand -> dw_kernel): the expand accumulates the same k16 chunks in
// order with the same mma instruction, applies bias/activation in fp32 and rounds pairs to the storage type exactly
// like the GEMM epilogue; the depthwise loop visits taps in the same order, skips the same out-of-image taps and uses
// the same fmaf chain and epilogue.  (Verified on all 840 crops by bench/gpu_fuse_check.sh.)
#include "kernels.cuh"
#include "gemm.cuh"

#include <algorithm>

namespace shv {
namespace {

template <typename T, int KS, int S>
__global__ void __launch_bounds__(256)
expand_dw_kernel(const T* __restrict__ X, const T* __restrict__ We, const float* __restrict__ be,
                 const float* __restrict__ wd, const float* __restrict__ bd, T* __restrict__ Dout, int H, int W, int Cin,
                 int Cexp, int Ho, int Wo, int act, int TH, int TW, int CC, int tiles_w) {
    extern __shared__ __align__(16) unsigned char sm[];
    constexpr int P = (KS - 1) / 2;
    const int f = blockIdx.y, tile = blockIdx.x, chunk = blockIdx.z;
    const int oh0 = (tile / tiles_w) * TH, ow0 = (tile % tiles_w) * TW;
    const int c0 = chunk * CC, cc = min(CC, Cexp - c0);
    const int IH = (TH - 1) * S + KS, IW = (TW - 1) * S + KS, NPX = IH * IW, NPXp = (NPX + 15) & ~15;
    const int ih0 = oh0 * S - P, iw0 = ow0 * S - P;
    const int Kp = (Cin + 15) & ~15, XP = Kp + 8, EP = CC + 8, kc8 = Kp / 8;
    T* xs = reinterpret_cast<T*>(sm);   // [NPXp][XP]  input halo tile
    T* ws = xs + NPXp * XP;             // [CC][XP]    expand weights of this channel chunk
    T* es = ws + CC * XP;               // [NPXp][EP]  expanded tile
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long fbase = (long long)f * H * W;
    for (int i = tid; i < NPXp * kc8; i += 256) {
        const int px = i / kc8, k8 = (i % kc8) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (px < NPX && k8 < Cin) {
            const int ih = ih0 + px / IW, iw = iw0 + px % IW;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                v = __ldg(reinterpret_cast<const uint4*>(X + (fbase + (long long)ih * W + iw) * Cin + k8));
        }
        *reinterpret_cast<uint4*>(xs + px * XP + k8) = v;
    }
    for (int i = tid; i < CC * kc8; i += 256) {
        const int c = i / kc8, k8 = (i % kc8) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (c < cc && k8 < Cin) v = __ldg(reinterpret_cast<const uint4*>(We + (long long)(c0 + c) * Cin + k8));
        *reinterpret_cast<uint4*>(ws + c * XP + k8) = v;
    }
    __syncthreads();
    // expand: warp w computes pixel m-tiles w, w+8, ... for all n8 tiles of the chunk
    const int NT8 = cc / 8, g = lane >> 2, c2 = (lane & 3) * 2;
    using P2 = Pair2<T>;
    for (int mt = warp; mt < NPXp / 16; mt += 8) {
        float acc[16][4];
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
        for (int ks = 0; ks < Kp / 16; ++ks) {
            uint32_t a[4];
            ldmatrix_x4(a, xs + (mt * 16 + (lane & 15)) * XP + ks * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int nt = 0; nt < 16; ++nt)
                if (nt < NT8) {
                    uint32_t b[2];
                    ldmatrix_x2(b, ws + (nt * 8 + (lane & 7)) * XP + ks * 16 + ((lane >> 3) & 1) * 8);
                    mma16816<T>(acc[nt], a, b);
                }
        }
#pragma unroll
        for (int nt = 0; nt < 16; ++nt)
            if (nt < NT8)
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const int px = mt * 16 + g + h2 * 8, col = nt * 8 + c2;
                    float v0 = apply_act(acc[nt][h2 * 2] + be[c0 + col], act);
                    float v1 = apply_act(acc[nt][h2 * 2 + 1] + be[c0 + col + 1], act);
                    *reinterpret_cast<typename P2::type*>(es + px * EP + col) = P2::make(v0, v1);
                }
    }
    __syncthreads();
    // depthwise from shared memory
    const int CG = cc / 8;
    for (int task = tid; task < TH * TW * CG; task += 256) {
        const int cg = task % CG, opx = task / CG, oy = opx / TW, ox = opx % TW;
        const int oh = oh0 + oy, ow = ow0 + ox;
        if (oh >= Ho || ow >= Wo) continue;
        const int cglob = c0 + cg * 8;
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
                load8(es + ((ih - ih0) * IW + (iw - iw0)) * EP + cg * 8, x);
                const float4 w0 = __ldg(reinterpret_cast<const float4*>(wd + (kh * KS + kw) * Cexp + cglob));
                const float4 w1 = __ldg(reinterpret_cast<const float4*>(wd + (kh * KS + kw) * Cexp + cglob + 4));
                acc[0] = fmaf(w0.x, x[0], acc[0]); acc[1] = fmaf(w0.y, x[1], acc[1]);
                acc[2] = fmaf(w0.z, x[2], acc[2]); acc[3] = fmaf(w0.w, x[3], acc[3]);
                acc[4] = fmaf(w1.x, x[4], acc[4]); acc[5] = fmaf(w1.y, x[5], acc[5]);
                acc[6] = fmaf(w1.z, x[6], acc[6]); acc[7] = fmaf(w1.w, x[7], acc[7]);
            }
        }
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = apply_act(acc[i] + bd[cglob + i], act);
        store8(Dout + ((long long)f * Ho * Wo + (long long)oh * Wo + ow) * Cexp + cglob, v);
    }
}

size_t fused_smem(int TH, int TW, int CC, int Cin, int KS, int S) {
    const int IH = (TH - 1) * S + KS, IW = (TW - 1) * S + KS, NPXp = (IH * IW + 15) & ~15;
    const int Kp = (Cin + 15) & ~15;
    return size_t(NPXp * (Kp + 8) + CC * (Kp + 8) + NPXp * (CC + 8)) * 2;
}

template <typename T, int KS, int S> void set_fused_attr() {
    SHV_CUDA_CHECK(cudaFuncSetAttribute(expand_dw_kernel<T, KS, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
}

}  // namespace

FusedPlan plan_expand_dw(int Cin, int Cexp, int KS, int S, int Ho, int Wo) {
    FusedPlan p;
    const int nchunks = (Cexp + 127) / 128;
    p.CC = 8 * ((Cexp + 8 * nchunks - 1) / (8 * nchunks));
    p.chunks = (Cexp + p.CC - 1) / p.CC;
    static const int cand[][2] = {{8, 16}, {4, 16}, {8, 8}, {4, 10}, {6, 10}, {4, 8}, {3, 10}, {2, 10}, {4, 4}, {2, 8}, {2, 4}, {1, 10}, {2, 2}, {1, 1}};
    for (const auto& c : cand) {
        const int th = std::min(c[0], Ho), tw = std::min(c[1], Wo);
        const size_t sm = fused_smem(th, tw, p.CC, Cin, KS, S);
        if (sm <= 48 * 1024) { p.TH = th; p.TW = tw; p.smem = sm; break; }
    }
    if (!p.TH) {
        for (const auto& c : cand) {
            const int th = std::min(c[0], Ho), tw = std::min(c[1], Wo);
            const size_t sm = fused_smem(th, tw, p.CC, Cin, KS, S);
            if (sm <= 96 * 1024) { p.TH = th; p.TW = tw; p.smem = sm; break; }
        }
    }
    if (!p.TH) throw std::runtime_error("no fused tile fits shared memory");
    p.tiles_w = (Wo + p.TW - 1) / p.TW;
    p.tiles = ((Ho + p.TH - 1) / p.TH) * p.tiles_w;
    return p;
}

void fused_init_attributes() {
    set_fused_attr<__nv_bfloat16, 3, 1>(); set_fused_attr<__nv_bfloat16, 3, 2>();
    set_fused_attr<__nv_bfloat16, 5, 1>(); set_fused_attr<__nv_bfloat16, 5, 2>();
    set_fused_attr<__half, 3, 1>(); set_fused_attr<__half, 3, 2>();
    set_fused_attr<__half, 5, 1>(); set_fused_attr<__half, 5, 2>();
}

template <typename T>
void launch_expand_dw(const FusedPlan& p, const T* X, const T* We, const float* be, const float* wd, const float* bd, T* D,
                      int F, int H, int W, int Cin, int Cexp, int KS, int S, int Ho, int Wo, int act, cudaStream_t s) {
    const dim3 grid(p.tiles, F, p.chunks);
#define SHV_FUSED(K, ST)                                                                                          \
    expand_dw_kernel<T, K, ST><<<grid, 256, p.smem, s>>>(X, We, be, wd, bd, D, H, W, Cin, Cexp, Ho, Wo, act, p.TH, p.TW, \
                                                          p.CC, p.tiles_w)
    if (KS == 3 && S == 1) SHV_FUSED(3, 1);
    else if (KS == 3 && S == 2) SHV_FUSED(3, 2);
    else if (KS == 5 && S == 1) SHV_FUSED(5, 1);
    else if (KS == 5 && S == 2) SHV_FUSED(5, 2);
    else throw std::runtime_error("unsupported fused depthwise kernel");
#undef SHV_FUSED
    SHV_CUDA_CHECK(cudaGetLastError());
}
template void launch_expand_dw<__nv_bfloat16>(const FusedPlan&, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*,
                                              const float*, __nv_bfloat16*, int, int, int, int, int, int, int, int, int, int,
                                              cudaStream_t);
template void launch_expand_dw<__half>(const FusedPlan&, const __half*, const __half*, const float*, const float*, const float*,
                                       __half*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);

}  // namespace shv
