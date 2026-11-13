// SHIELD-V CUDA engine: kernel launchers (CNN: cnn.cu, sequence: seq.cu).
#pragma once
#include "common.cuh"
#include "model.hpp"

namespace shv {

// ---------------------------------------------------------------- CNN (NHWC, storage type T, fp32 accumulation)
// uint8 RGB [F,180,320,3] -> T [F,90,160,16]: LUT normalisation (exactly prepare_clips), 3x3 s2 conv, folded BN,
// hardswish.  lut: [3][256] fp32, w: [3][3][3][16] (kh,kw,ci,co), b: [16].
template <typename T>
void launch_stem(const uint8_t* in, T* out, const float* lut, const float* w, const float* b, int F, cudaStream_t s);
// depthwise KxK conv, stride S, pad (K-1)/2, folded BN + activation. w: [K*K][C], C % 8 == 0.
template <typename T>
void launch_dw(const T* in, T* out, const float* w, const float* b, int F, int H, int W, int C, int K, int S,
               int Ho, int Wo, int act, cudaStream_t s);
// SqueezeExcitation pooling: pooled[f][c] = mean_hw x[f][hw][c] (fixed-order reduction, fp32).  The two SE
// 1x1 convs then run as fp32 GEMMs over all frames (relu / hardsigmoid epilogues).
// part: [F][se_splits(HW)][C] scratch.
int se_splits(int HW);
template <typename T> void launch_se_pool(const T* x, float* part, float* pooled, int F, int HW, int C, cudaStream_t s);
// Fused expand (1x1 conv, folded BN, act) -> depthwise for bf16/fp16 (fused.cu): tile plan chosen from the layer only.
struct FusedPlan { int TH = 0, TW = 0, CC = 0, chunks = 0, tiles = 0, tiles_w = 0; size_t smem = 0; };
FusedPlan plan_expand_dw(int Cin, int Cexp, int KS, int S, int Ho, int Wo);
void fused_init_attributes();
template <typename T>
void launch_expand_dw(const FusedPlan& p, const T* X, const T* We, const float* be, const float* wd, const float* bd, T* D,
                      int F, int H, int W, int Cin, int Cexp, int KS, int S, int Ho, int Wo, int act, cudaStream_t s);
// adaptive_avg_pool2d([F,6,10,960] NHWC -> 4x4), output T [F*16, 960] (token = row*4 + col)
template <typename T> void launch_pool(const T* in, T* out, int F, cudaStream_t s);

// ---------------------------------------------------------------- sequence (residual stream fp32)
// h[b*321 + t] = tokens(b, t/16, t%16) + pos[t] for t < 320, readout for t == 320.  tok_ptrs: [B*20] device
// pointers to 16x384 fp32 token blocks.
void launch_assemble(const float* const* tok_ptrs, const float* pos, const float* readout, float* h, int B, cudaStream_t s);
// out[r] = (x[r] * (1/sqrt(mean(x[r]^2) + eps))) * w, rows of 384.
template <typename T>
void launch_rmsnorm(const float* x, long long ldx, const float* w, T* out, long long ldo, int rows, cudaStream_t s);
// GDN: causal short conv (k4, bias) + SiLU on q|k|v, l2norm(q) * 64^-0.5, l2norm(k), decay = exp(-exp(A_log) *
// softplus(a + dt_bias)), beta = sigmoid(b).  P: [rows][2320] fused projection output.
void launch_gdn_prep(const float* P, const float* conv_w, const float* conv_b, const float* a_log, const float* dt_bias,
                     float* Q, float* Kn, float* Vc, float* dec, float* beta, int B, cudaStream_t s);
// gated delta rule, exact recurrence from zero state, fp32: O [B*321][768]
void launch_delta(const float* Q, const float* Kn, const float* Vc, const float* dec, const float* beta, float* O, int B,
                  cudaStream_t s);
// out = ((o * rstd) * w) * silu(gate) per (row, head), gate = P[:, 1536 + h*128 ...]
template <typename T>
void launch_gated_norm(const float* O, const float* P, const float* w, T* out, int rows, cudaStream_t s);
// SISA attention of the last row (token 320) per (window, head): scans in double, rotation, scores, softmax, P V.
void launch_sisa_attn(const float* S, const float* QC, const float* lambda_raw, float* Y, int B, cudaStream_t s);
// final RMSNorm + Linear(384,128) + SiLU + Linear(128,1), fp32
// pool[slots[f]] = tokens[f] for slots[f] >= 0 (device token slots, 6144 floats each)
void launch_store_tokens(const float* tokens, const int32_t* slots, float* pool, int F, cudaStream_t s);
void launch_head(const float* X, const float* norm_w, const float* w0, const float* b0, const float* w2, const float* b2,
                 float* logits, int B, cudaStream_t s);

}  // namespace shv
