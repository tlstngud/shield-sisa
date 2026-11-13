// SHIELD-V CUDA engine: sequence kernels (GDN-A, GDN-B x2, last-row SISA, head).  Residual stream and all
// non-GEMM arithmetic in fp32; SISA cumulative sums in double, rounded once per element (torch CPU semantics).
#include "kernels.cuh"
#include "gemm.cuh"

namespace shv {
namespace {

template <typename T> __device__ __forceinline__ void store4(T* p, float a, float b, float c, float d);
template <> __device__ __forceinline__ void store4<float>(float* p, float a, float b, float c, float d) {
    *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}
template <typename Hf> __device__ __forceinline__ void store4_half(Hf* p, float a, float b, float c, float d) {
    uint2 u;
    Hf* h = reinterpret_cast<Hf*>(&u);
    h[0] = from_f<Hf>(a); h[1] = from_f<Hf>(b); h[2] = from_f<Hf>(c); h[3] = from_f<Hf>(d);
    *reinterpret_cast<uint2*>(p) = u;
}
template <> __device__ __forceinline__ void store4<__nv_bfloat16>(__nv_bfloat16* p, float a, float b, float c, float d) {
    store4_half(p, a, b, c, d);
}
template <> __device__ __forceinline__ void store4<__half>(__half* p, float a, float b, float c, float d) {
    store4_half(p, a, b, c, d);
}

// ------------------------------------------------------------------------------------------------ assemble
__global__ void __launch_bounds__(96)
assemble_kernel(const float* const* __restrict__ tok_ptrs, const float* __restrict__ pos,
                const float* __restrict__ readout, float* __restrict__ h) {
    const long long row = blockIdx.x;
    const int b = int(row / NTOK), t = int(row % NTOK), j = threadIdx.x * 4;
    float4 v;
    if (t == NTOK - 1) {
        v = *reinterpret_cast<const float4*>(readout + j);
    } else {
        const float4 a = *reinterpret_cast<const float4*>(tok_ptrs[b * FRAMES + t / 16] + (t % 16) * D + j);
        const float4 p = *reinterpret_cast<const float4*>(pos + t * D + j);
        v = make_float4(a.x + p.x, a.y + p.y, a.z + p.z, a.w + p.w);
    }
    *reinterpret_cast<float4*>(h + row * D + j) = v;
}

// ------------------------------------------------------------------------------------------------ RMSNorm
template <typename T>
__global__ void __launch_bounds__(256)
rmsnorm_kernel(const float* __restrict__ x, long long ldx, const float* __restrict__ w, T* __restrict__ out,
               long long ldo, int rows) {
    const int row = int((blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* xr = x + row * ldx;
    float v[12];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float4 a = *reinterpret_cast<const float4*>(xr + i * 128 + lane * 4);
        v[i * 4 + 0] = a.x; v[i * 4 + 1] = a.y; v[i * 4 + 2] = a.z; v[i * 4 + 3] = a.w;
    }
#pragma unroll
    for (int i = 0; i < 12; ++i) ss += v[i] * v[i];
    ss = warp_sum(ss);
    const float r = 1.0f / sqrtf(ss / float(D) + 1e-6f);
    T* o = out + row * ldo;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int c = i * 128 + lane * 4;
        store4<T>(o + c, (v[i * 4 + 0] * r) * w[c + 0], (v[i * 4 + 1] * r) * w[c + 1], (v[i * 4 + 2] * r) * w[c + 2],
                  (v[i * 4 + 3] * r) * w[c + 3]);
    }
}

// ------------------------------------------------------------------------------------------------ GDN prep
__global__ void __launch_bounds__(384)
gdn_prep_kernel(const float* __restrict__ P, const float* __restrict__ cw, const float* __restrict__ cb,
                const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ Q,
                float* __restrict__ Kn, float* __restrict__ Vc, float* __restrict__ dec, float* __restrict__ beta) {
    const long long row = blockIdx.x;
    const int t = int(row % NTOK), c = threadIdx.x, lane = c & 31, warp = c >> 5;
    auto conv = [&](int col, int ch) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int dt = 3 - j;
            if (t - dt >= 0) acc = fmaf(cw[ch * 4 + j], P[(row - dt) * GLD + col], acc);
        }
        return silu(acc + cb[ch]);
    };
    const float q = conv(GQ + c, c), k = conv(GK + c, 384 + c);
    Vc[row * VD + c] = conv(GV + c, 768 + c);
    Vc[row * VD + 384 + c] = conv(GV + 384 + c, 1152 + c);
    __shared__ float red[2][12];
    const float sq = warp_sum(q * q), sk = warp_sum(k * k);
    if (lane == 0) { red[0][warp] = sq; red[1][warp] = sk; }
    __syncthreads();
    const int hh = c >> 6;
    const float tq = red[0][2 * hh] + red[0][2 * hh + 1], tk = red[1][2 * hh] + red[1][2 * hh + 1];
    Q[row * D + c] = (q * (1.0f / sqrtf(tq + 1e-6f))) * 0.125f;
    Kn[row * D + c] = k * (1.0f / sqrtf(tk + 1e-6f));
    if (c < H) {
        const float* p = P + row * GLD;
        const float g = -expf(a_log[c]) * softplusf_(p[GA + c] + dt_bias[c]);
        dec[row * 8 + c] = expf(g);
        beta[row * 8 + c] = sigmoidf_(p[GB + c]);
    }
}

// ------------------------------------------------------------------------------------------------ delta rule
// grid (B*H, 4): each CTA owns 32 value columns of one head; thread = (column, K quarter) holds 16 state entries.
// Per token (oracle order): S = a*S; pred = S^T k; u = beta*(v - pred); S += k u^T; o = S^T q.
constexpr int DT_TC = 32;
__global__ void __launch_bounds__(128)
delta_kernel(const float* __restrict__ Q, const float* __restrict__ Kn, const float* __restrict__ Vc,
             const float* __restrict__ dec, const float* __restrict__ beta, float* __restrict__ O) {
    const int b = blockIdx.x / H, h = blockIdx.x % H, vbase = blockIdx.y * 32;
    const int tid = threadIdx.x, kq = tid & 3, cl = tid >> 2;
    __shared__ __align__(16) float sk[2][DT_TC][DK];
    __shared__ __align__(16) float sq[2][DT_TC][DK];
    __shared__ __align__(16) float sv[2][DT_TC][32];
    __shared__ float sa[2][DT_TC], sbt[2][DT_TC];
    __shared__ __align__(16) float so[DT_TC][32];
    const long long row0 = (long long)b * NTOK;
    auto load_chunk = [&](int buf, int t0) {
        for (int i = tid; i < DT_TC * 16; i += 128) {
            const int tt = i >> 4, c4 = (i & 15) * 4, t = t0 + tt;
            const bool ok = t < NTOK;
            cp_async16(&sk[buf][tt][c4], ok ? Kn + (row0 + t) * D + h * DK + c4 : Kn, ok);
            cp_async16(&sq[buf][tt][c4], ok ? Q + (row0 + t) * D + h * DK + c4 : Q, ok);
        }
        for (int i = tid; i < DT_TC * 8; i += 128) {
            const int tt = i >> 3, c4 = (i & 7) * 4, t = t0 + tt;
            const bool ok = t < NTOK;
            cp_async16(&sv[buf][tt][c4], ok ? Vc + (row0 + t) * VD + h * DV + vbase + c4 : Vc, ok);
        }
        if (tid < DT_TC) {
            const int t = t0 + tid;
            sa[buf][tid] = t < NTOK ? dec[(row0 + t) * 8 + h] : 0.0f;
            sbt[buf][tid] = t < NTOK ? beta[(row0 + t) * 8 + h] : 0.0f;
        }
    };
    float S[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) S[j] = 0.0f;
    constexpr int NCH = (NTOK + DT_TC - 1) / DT_TC;
    load_chunk(0, 0);
    cp_async_commit();
    for (int ch = 0; ch < NCH; ++ch) {
        const int buf = ch & 1;
        if (ch + 1 < NCH) load_chunk(buf ^ 1, (ch + 1) * DT_TC);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const int nt = min(DT_TC, NTOK - ch * DT_TC);
        auto step = [&](int i) {
            const float a = sa[buf][i], bt = sbt[buf][i];
            float k[16], q[16];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 kv = *reinterpret_cast<const float4*>(&sk[buf][i][kq * 16 + j * 4]);
                const float4 qv = *reinterpret_cast<const float4*>(&sq[buf][i][kq * 16 + j * 4]);
                k[j * 4 + 0] = kv.x; k[j * 4 + 1] = kv.y; k[j * 4 + 2] = kv.z; k[j * 4 + 3] = kv.w;
                q[j * 4 + 0] = qv.x; q[j * 4 + 1] = qv.y; q[j * 4 + 2] = qv.z; q[j * 4 + 3] = qv.w;
            }
            const float v = sv[buf][i][cl];
#pragma unroll
            for (int j = 0; j < 16; ++j) S[j] *= a;
            float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) p[j & 3] = fmaf(S[j], k[j], p[j & 3]);
            float pred = (p[0] + p[1]) + (p[2] + p[3]);
            pred += __shfl_xor_sync(0xffffffffu, pred, 1);
            pred += __shfl_xor_sync(0xffffffffu, pred, 2);
            const float u = (v - pred) * bt;
#pragma unroll
            for (int j = 0; j < 16; ++j) S[j] = fmaf(k[j], u, S[j]);
            float o4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) o4[j & 3] = fmaf(S[j], q[j], o4[j & 3]);
            float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            if (kq == 0) so[i][cl] = o;
        };
        if (nt == DT_TC) {
#pragma unroll 4
            for (int i = 0; i < DT_TC; ++i) step(i);
        } else {
            for (int i = 0; i < nt; ++i) step(i);
        }
        __syncthreads();
        for (int i = tid; i < nt * 8; i += 128) {
            const int tt = i >> 3, c4 = (i & 7) * 4;
            *reinterpret_cast<float4*>(O + (row0 + ch * DT_TC + tt) * VD + h * DV + vbase + c4) =
                *reinterpret_cast<const float4*>(&so[tt][c4]);
        }
    }
}

// ------------------------------------------------------------------------------------------------ gated norm
template <typename T>
__global__ void __launch_bounds__(256)
gated_norm_kernel(const float* __restrict__ O, const float* __restrict__ P, const float* __restrict__ w,
                  T* __restrict__ out, int rows) {
    const long long wid = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= (long long)rows * H) return;
    const long long row = wid / H;
    const int h = int(wid % H), c = lane * 4;
    const float4 o = *reinterpret_cast<const float4*>(O + row * VD + h * DV + c);
    const float4 g = *reinterpret_cast<const float4*>(P + row * GLD + GG + h * DV + c);
    float ss = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
    ss = warp_sum(ss);
    const float rstd = 1.0f / sqrtf(ss / float(DV) + 1e-6f);
    store4<T>(out + row * VD + h * DV + c, ((o.x * rstd) * w[c + 0]) * silu(g.x), ((o.y * rstd) * w[c + 1]) * silu(g.y),
              ((o.z * rstd) * w[c + 2]) * silu(g.z), ((o.w * rstd) * w[c + 3]) * silu(g.w));
}

// ------------------------------------------------------------------------------------------------ SISA last row
constexpr int SA_THREADS = 256;
__global__ void __launch_bounds__(SA_THREADS)
sisa_attn_kernel(const float* __restrict__ S, const float* __restrict__ QC, const float* __restrict__ lambda_raw,
                 float* __restrict__ Y) {
    const int b = blockIdx.x / H, h = blockIdx.x % H, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* S0 = S + (long long)b * NTOK * SLD;
    __shared__ float raw[NTOK][18];   // alpha, forget, theta[16] of this head, then their prefix sums
    __shared__ float sscore[NTOK];
    __shared__ float sq[DK], sc[DSTATE], scr[DSTATE];
    __shared__ float red[SA_THREADS / 32];
    __shared__ float yred[4][DK];
    // phase 1 (parallel): per-token terms -softplus(alpha), logsigmoid(forget), theta (fp32, as the reference)
    for (int i = tid; i < NTOK * 18; i += SA_THREADS) {
        const int t = i / 18, c = i % 18;
        const int col = c == 0 ? SA + h : c == 1 ? SF + h : ST_ + h * NPAIR + (c - 2);
        const float x = S0[(long long)t * SLD + col];
        raw[t][c] = c == 0 ? -softplusf_(x) : c == 1 ? logsigmoidf_(x) : x;
    }
    if (tid < DK) sq[tid] = QC[(long long)b * SQN + SQ + h * DK + tid];
    else if (tid < DK + DSTATE) sc[tid - DK] = QC[(long long)b * SQN + SC + h * DSTATE + (tid - DK)];
    __syncthreads();
    // phase 2: sequential prefix sums in double, rounded once per element (torch CPU cumsum semantics)
    if (tid < 18) {
        double acc = 0.0;
#pragma unroll 8
        for (int t = 0; t < NTOK; ++t) {
            acc += double(raw[t][tid]);
            raw[t][tid] = float(acc);
        }
    }
    __syncthreads();
    if (tid < NPAIR) {
        float si, co;
        sincosf(raw[NTOK - 1][2 + tid], &si, &co);
        const float e = sc[2 * tid], o = sc[2 * tid + 1];
        scr[2 * tid] = e * co - o * si;
        scr[2 * tid + 1] = e * si + o * co;
    }
    __syncthreads();
    const float lam = softplusf_(lambda_raw[h]);
    const float g_last = raw[NTOK - 1][0], f_last = raw[NTOK - 1][1];
    float lmax = -INFINITY;
    for (int t = tid; t < NTOK; t += SA_THREADS) {
        const float* kr = S0 + (long long)t * SLD + SK + h * DK;
        float content = 0.0f;
#pragma unroll 16
        for (int j = 0; j < DK; ++j) content = fmaf(sq[j], kr[j], content);
        content = content * 0.125f;
        const float* br = S0 + (long long)t * SLD + SB + h * DSTATE;
        float ssm = 0.0f;
#pragma unroll 4
        for (int p = 0; p < NPAIR; ++p) {
            float si, co;
            sincosf(raw[t][2 + p], &si, &co);
            const float e = br[2 * p], o = br[2 * p + 1];
            const float be = e * co - o * si, bo = e * si + o * co;
            ssm = fmaf(scr[2 * p], be, ssm);
            ssm = fmaf(scr[2 * p + 1], bo, ssm);
        }
        float s = content + (lam * expf(g_last - raw[t][0])) * ssm;
        s = s + (f_last - raw[t][1]);
        sscore[t] = s;
        lmax = fmaxf(lmax, s);
    }
    lmax = warp_max(lmax);
    if (lane == 0) red[warp] = lmax;
    __syncthreads();
    float mx = red[0];
    for (int i = 1; i < SA_THREADS / 32; ++i) mx = fmaxf(mx, red[i]);
    __syncthreads();
    float lsum = 0.0f;
    for (int t = tid; t < NTOK; t += SA_THREADS) {
        const float e = expf(sscore[t] - mx);
        sscore[t] = e;
        lsum += e;
    }
    lsum = warp_sum(lsum);
    if (lane == 0) red[warp] = lsum;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < SA_THREADS / 32; ++i) sum += red[i];
    const float inv = 1.0f / sum;
    const int j = tid & (DK - 1), grp = tid >> 6;
    float y = 0.0f;
#pragma unroll 8
    for (int t = grp; t < NTOK; t += 4) y = fmaf(sscore[t] * inv, S0[(long long)t * SLD + SV + h * DK + j], y);
    yred[grp][j] = y;
    __syncthreads();
    if (tid < DK) Y[(long long)b * D + h * DK + tid] = (yred[0][tid] + yred[1][tid]) + (yred[2][tid] + yred[3][tid]);
}

// ------------------------------------------------------------------------------------------------ head
__global__ void __launch_bounds__(128)
head_kernel(const float* __restrict__ X, const float* __restrict__ nw, const float* __restrict__ w0t /*[D][128]*/,
            const float* __restrict__ b0, const float* __restrict__ w2, const float* __restrict__ b2,
            float* __restrict__ logits) {
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float z[D];
    __shared__ float red[4];
    const float* x = X + (long long)b * D;
    const float v0 = x[tid], v1 = x[tid + 128], v2 = x[tid + 256];
    float ss = (v0 * v0 + v1 * v1) + v2 * v2;
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    const float tot = (red[0] + red[1]) + (red[2] + red[3]);
    const float r = 1.0f / sqrtf(tot / float(D) + 1e-6f);
    z[tid] = (v0 * r) * nw[tid];
    z[tid + 128] = (v1 * r) * nw[tid + 128];
    z[tid + 256] = (v2 * r) * nw[tid + 256];
    __syncthreads();
    float acc = 0.0f;
    for (int jj = 0; jj < D; ++jj) acc = fmaf(w0t[jj * 128 + tid], z[jj], acc);
    const float hid = silu(acc + b0[tid]);
    float prod = warp_sum(w2[tid] * hid);
    __syncthreads();
    if (lane == 0) red[warp] = prod;
    __syncthreads();
    if (tid == 0) logits[b] = ((red[0] + red[1]) + (red[2] + red[3])) + b2[0];
}

__global__ void __launch_bounds__(256)
store_tokens_kernel(const float* __restrict__ src, const int32_t* __restrict__ slots, float* __restrict__ pool) {
    const int f = blockIdx.x, s = slots[f];
    if (s < 0) return;
    const float4* a = reinterpret_cast<const float4*>(src + (long long)f * 16 * D);
    float4* b = reinterpret_cast<float4*>(pool + (long long)s * 16 * D);
    for (int i = threadIdx.x; i < 16 * D / 4; i += blockDim.x) b[i] = a[i];
}

}  // namespace

void launch_store_tokens(const float* tokens, const int32_t* slots, float* pool, int F, cudaStream_t s) {
    store_tokens_kernel<<<F, 256, 0, s>>>(tokens, slots, pool);
}
void launch_assemble(const float* const* tok_ptrs, const float* pos, const float* readout, float* h, int B,
                     cudaStream_t s) {
    assemble_kernel<<<B * NTOK, 96, 0, s>>>(tok_ptrs, pos, readout, h);
}
template <typename T>
void launch_rmsnorm(const float* x, long long ldx, const float* w, T* out, long long ldo, int rows, cudaStream_t s) {
    rmsnorm_kernel<T><<<ceil_div((long long)rows * 32, 256), 256, 0, s>>>(x, ldx, w, out, ldo, rows);
}
void launch_gdn_prep(const float* P, const float* conv_w, const float* conv_b, const float* a_log, const float* dt_bias,
                     float* Q, float* Kn, float* Vc, float* dec, float* beta, int B, cudaStream_t s) {
    gdn_prep_kernel<<<B * NTOK, 384, 0, s>>>(P, conv_w, conv_b, a_log, dt_bias, Q, Kn, Vc, dec, beta);
}
void launch_delta(const float* Q, const float* Kn, const float* Vc, const float* dec, const float* beta, float* O, int B,
                  cudaStream_t s) {
    delta_kernel<<<dim3(B * H, DV / 32), 128, 0, s>>>(Q, Kn, Vc, dec, beta, O);
}
template <typename T>
void launch_gated_norm(const float* O, const float* P, const float* w, T* out, int rows, cudaStream_t s) {
    gated_norm_kernel<T><<<ceil_div((long long)rows * H * 32, 256), 256, 0, s>>>(O, P, w, out, rows);
}
void launch_sisa_attn(const float* S, const float* QC, const float* lambda_raw, float* Y, int B, cudaStream_t s) {
    sisa_attn_kernel<<<B * H, SA_THREADS, 0, s>>>(S, QC, lambda_raw, Y);
}
void launch_head(const float* X, const float* norm_w, const float* w0t, const float* b0, const float* w2, const float* b2,
                 float* logits, int B, cudaStream_t s) {
    head_kernel<<<B, 128, 0, s>>>(X, norm_w, w0t, b0, w2, b2, logits);
}

#define SHV_INST_SEQ(T)                                                                                   \
    template void launch_rmsnorm<T>(const float*, long long, const float*, T*, long long, int, cudaStream_t); \
    template void launch_gated_norm<T>(const float*, const float*, const float*, T*, int, cudaStream_t);
SHV_INST_SEQ(float)
SHV_INST_SEQ(__nv_bfloat16)
SHV_INST_SEQ(__half)

}  // namespace shv
