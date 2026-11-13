// SHIELD-V CPU engine: plain-data model description shared by the runtime (baseline ISA) and the
// per-ISA kernel translation units (isa_*.cpp).  POD only: no member functions, no std templates,
// so that nothing in here can be emitted as an ODR-merged inline function compiled for AVX-512.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace shv {

enum { SHV_H = 180, SHV_W = 320, SHV_FRAMES = 20, SHV_TOK = 16, SHV_D = 384, SHV_T = 321 };
enum { SHV_HEADS = 6, SHV_DK = 64, SHV_DV = 128, SHV_VD = 768, SHV_FF = 960, SHV_DSTATE = 32 };
// fused GDN in-projection column layout (row stride GDN_P)
enum { GDN_Q = 0, GDN_K = 384, GDN_V = 768, GDN_G = 1536, GDN_A = 2304, GDN_B = 2310, GDN_N = 2316, GDN_P = 2320 };
// fused SISA in-projection column layout (row stride SISA_P)
enum { SISA_K = 0, SISA_V = 384, SISA_B = 768, SISA_TH = 960, SISA_AL = 1056, SISA_FG = 1062, SISA_N = 1068, SISA_P = 1072 };

enum Act { ACT_NONE = 0, ACT_RELU = 1, ACT_HS = 2 };

// Numeric format of a packed linear layer.  FP32 is the exact tier.
enum Fmt { FMT_F32 = 0, FMT_BF16 = 1, FMT_INT8 = 2 };

// y[m][n] = sum_k x[m][k] * W[n][k] (+ bias[n]); W packed in 16-column panels:
//   f32:  wp[(p*K + k)*16 + lane]           = W[p*16+lane][k]       (zero for padded columns)
//   bf16: wb[((p*KP + kk)*16 + lane)*2 + e] = bf16(W[p*16+lane][2kk+e]), KP = ceil(K/2), zero padded
struct Linear {
    const float* wp;       // FP32 panels (always present: exact tier and fallback)
    const uint16_t* wb;    // BF16 pair-interleaved panels (only when fmt == FMT_BF16)
    const int8_t* wq;      // INT8 panels [Np/16][ceil(K/4)][16][4] (only when fmt == FMT_INT8)
    const float* wscale;   // [Np] per-output-channel INT8 weight scale
    const int32_t* wcorr;  // [Np] 128 * sum_k wq (zero-point correction)
    const float* bias;     // [Np] or nullptr (padded entries zero)
    int K, N, Np;          // real input width, real output width, output width padded to 16
    int fmt;
};

struct Block {
    int H, W, Ho, Wo;                 // input / output spatial size
    int cin, cinp, cexp, cexpp, cout, coutp;  // real and 16-padded channel counts
    int k, s, p, act;                 // depthwise kernel, stride, padding, activation (expand + dw)
    int has_expand, has_se, residual;
    int full_solo, full_team;         // use the full-map expansion path (solo item / team item)
    Linear expand;                    // cin -> cexp, BN folded (bias), act
    const float* dw_w;                // [k*k][cexpp], BN folded
    const float* dw_b;                // [cexpp]
    int csq, csqp;                    // SE squeeze width
    Linear se1, se2;                  // cexp -> csq (ReLU), csq -> cexp (hardsigmoid), with bias
    Linear project;                   // cexp -> cout, BN folded, no act
};

struct Cnn {
    float lut[3][256];                // uint8 -> ((v/255) - mean_c) / std_c, float32 exactly as prepare_clips
    const float* stem_w;              // [27 taps (dy,dx,c)][16]
    const float* stem_b;              // [16]
    int nblocks;
    Block blocks[15];
    Linear last;                      // 160 -> 960, BN folded, hardswish
    Linear proj;                      // 960 -> 384, no bias
    void (*dump)(void* ctx, const char* name, const float* data, size_t n);  // debug (SHV_CPU_DUMP_DIR) or null
    void* dump_ctx;
};

struct Gdn {
    const float* norm_w;              // stage RMSNorm (eps 1e-6)
    const float* ffn_norm_w;
    Linear in;                        // fused q,k,v,g,a,b  (N = 2316, Np = 2320), no bias
    const float* conv_w;              // [4][1536] tap-major: q(384) k(384) v(768) channels
    const float* conv_b;              // [1536]
    float neg_rate[SHV_HEADS];        // -exp(A_log)
    float dt_bias[SHV_HEADS];
    const float* o_norm_w;            // [128]
    Linear out;                       // o_proj 768 -> 384
    Linear gu;                        // gate/up interleaved per 16-column panel (N = Np = 1920)
    Linear down;                      // 960 -> 384
    int repeats;
};

struct Sisa {
    const float* attn_norm_w;
    const float* ffn_norm_w;
    Linear in;                        // fused K,V,B,theta,alpha(+bias),forget(+bias), N = 1068, Np = 1072
    Linear q;                         // W_Q (last row only)
    Linear c;                         // W_C (last row only)
    Linear o;                         // W_O
    float lam[SHV_HEADS];             // softplus(lambda_raw), float32
    Linear gu;                        // interleaved gate/up
    Linear down;
};

struct Seq {
    const float* pos;                 // [320][384] = (time[t] + row[r]) + col[c]
    const float* readout;             // [384]
    Gdn gdn[2];
    Sisa sisa;
    const float* final_norm_w;
    Linear head0;                     // 384 -> 128 (+bias), SiLU
    const float* head2_w;             // [128]
    float head2_b;
    void (*dump)(void* ctx, const char* name, const float* data, size_t n);
    void* dump_ctx;
};

// A team of threads cooperating on one frame or one window.  barrier() synchronises the team.
struct Team {
    int rank, size;
    void (*barrier)(void* ctx);
    void* ctx;
};

// Team-shared buffers of one frame (sized by the runtime for the network above).
struct FrameBuf {
    float* act[2];        // block input / output ping-pong (max H*W*Cp floats)
    float* dwfull;        // full depthwise output Ho*Wo*cexpp (SE blocks and the full-map path)
    float* xfull;         // full-map path: zero-bordered expanded input map
    float* lastmap;       // [60][960]
};
// Thread-private CNN scratch.
struct CnnScratch {
    float* ring;          // expanded-row ring
    float* strip;         // depthwise strip output (non-SE blocks)
    float* zero;          // a zero row (>= max padded row width)
    float* stemrows;      // 3 normalised padded input rows
    float* vec;           // SE / pooling vectors (>= 4*1024 floats)
    size_t ring_floats, strip_floats;
    int strip_rows;       // preferred output rows per strip (tuning)
    uint32_t* a16;        // bf16 activation pairs / int8 groups for reduced-precision GEMMs (or null)
    size_t a16_cap;       // capacity in 32-bit units
    float* ascale;        // INT8 per-row activation scales (4096) or null
};

// Team-shared buffers of one window.
struct SeqBuf {
    float* h;     // [321][384]
    float* hn;    // [321][384]
    float* P;     // [321][2320]
    float* O;     // [321][768]
    float* F;     // [321][960]  (SwiGLU product)
    float* vec;   // small vectors for the last row (>= 8192 floats)
    double* dvec; // per-head double accumulators
};
// Thread-private sequence scratch.
struct SeqScratch {
    float* state; // [64][128]
    float* rows;  // [321] scores etc. (>= 2048 floats)
    uint32_t* a16;   // bf16 activation pairs for BF16 GEMMs (or null)
    size_t a16_cap;
};

// Entry points exported by each ISA translation unit.
struct IsaOps {
    const char* name;
    void (*cnn_frame)(const Cnn* net, const uint8_t* rgb, float* tokens, const Team* team, FrameBuf* fb,
                      CnnScratch* sc);
    void (*seq_window)(const Seq* net, const float* const* frame_tokens, float* logit, const Team* team,
                       SeqBuf* sb, SeqScratch* sc);
    // building blocks exported for unit tests / benchmarks
    void (*gemm)(const float* A, int lda, int M, const Linear* L, int p0, int p1, float* C, int ldc, int epi,
                 const float* res, int ldr);
    // profile builds: read and reset per-stage nanoseconds / counts (zeros otherwise)
    void (*prof_get)(double* ns, long* n, int len);
};

enum Epi {
    EPI_STORE = 0,      // C = acc (+ bias if present)
    EPI_RELU = 1,       // C = relu(acc + bias)
    EPI_HS = 2,         // C = hardswish(acc + bias)
    EPI_RES = 3,        // C = (acc + bias) + res
    EPI_ACC = 4,        // C = C + acc
    EPI_ACC_DIV2 = 5,   // C = C + acc / sqrt(2)f
    EPI_SWIGLU = 6,     // panels (2i, 2i+1) = (gate, up): C[16i..] = silu(gate) * up
    EPI_HSIG = 7,       // C = hardsigmoid(acc + bias)
    EPI_SILU = 8        // C = silu(acc + bias)
};

}  // namespace shv

extern "C" {
const shv::IsaOps* shv_isa_ops_avx512(void);
const shv::IsaOps* shv_isa_ops_avx2(void);
const shv::IsaOps* shv_isa_ops_scalar(void);
}
