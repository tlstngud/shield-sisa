// SHIELD-V CUDA engine core: persistent device weights, activation buffers, forward passes, CUDA graphs.
#include "engine.hpp"
#include "gemm.cuh"
#include "kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

namespace shv {

const char* precision_name(Precision p) {
    switch (p) {
    case Precision::FP32: return "fp32";
    case Precision::BF16: return "bf16";
    case Precision::FP16: return "fp16";
    }
    return "?";
}
bool parse_precision(const std::string& s, Precision* p) {
    if (s == "fp32") { *p = Precision::FP32; return true; }
    if (s == "bf16") { *p = Precision::BF16; return true; }
    if (s == "fp16") { *p = Precision::FP16; return true; }
    return false;
}

static std::vector<int> make_buckets(int maxv) {
    static const int base[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256,
                               320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048};
    std::vector<int> b;
    for (int v : base)
        if (v < maxv) b.push_back(v);
    b.push_back(maxv);
    return b;
}
static int bucket_of(const std::vector<int>& b, int n) {
    for (int v : b)
        if (v >= n) return v;
    throw std::runtime_error("batch larger than the configured maximum");
}
int Engine::frame_bucket(int F) const { return cfg_.use_graphs ? bucket_of(fbuckets_, F) : F; }
int Engine::window_bucket(int B) const { return cfg_.use_graphs ? bucket_of(wbuckets_, B) : B; }

namespace {

// ------------------------------------------------------------------------------------------------ device arena
struct Arena {
    std::vector<uint8_t> host;
    template <typename U> size_t add(const std::vector<float>& v) {
        size_t off = (host.size() + 255) & ~size_t(255);
        host.resize(off + v.size() * sizeof(U));
        U* dst = reinterpret_cast<U*>(host.data() + off);
        for (size_t i = 0; i < v.size(); ++i) dst[i] = conv<U>(v[i]);
        return off;
    }
    template <typename U> static U conv(float x);
};
template <> float Arena::conv<float>(float x) { return x; }
template <> __nv_bfloat16 Arena::conv<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
template <> __half Arena::conv<__half>(float x) { return __float2half_rn(x); }

struct DevBuf {
    void* p = nullptr;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { if (p) cudaFree(p); }
    void alloc(size_t bytes) { SHV_CUDA_CHECK(cudaMalloc(&p, bytes)); SHV_CUDA_CHECK(cudaMemset(p, 0, bytes)); }
};

std::vector<float> norm_lut() {
    // exactly prepare_clips: x.float() / 255, then (x - mean) / std with float32 mean/std
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> lut(768);
    for (int c = 0; c < 3; ++c)
        for (int u = 0; u < 256; ++u) {
            volatile float x = float(u) / 255.0f;
            volatile float y = x - mean[c];
            lut[size_t(c) * 256 + u] = y / stdv[c];
        }
    return lut;
}

template <typename T>
class EngineT final : public Engine {
public:
    EngineT(const HostModel& hm, Precision p, const EngineConfig& cfg) {
        prec_ = p;
        cfg_ = cfg;
        if (cfg_.max_frames < 1 || cfg_.max_windows < 1 || cfg_.slots < 1 || cfg_.slots > 4)
            throw std::runtime_error("invalid engine configuration");
        fbuckets_ = make_buckets(cfg_.max_frames);
        wbuckets_ = make_buckets(cfg_.max_windows);
        gemm_init_attributes();
        if (!std::is_same<T, float>::value) {
            fused_init_attributes();
            // Fusion is bitwise neutral, so it is a pure speed choice: measured (docs/PERFORMANCE.md) to win only for the
            // stride-2 blocks with a single <=128-channel chunk and a thin input (features 2 and 4); elsewhere the halo
            // recompute and small tiles lose against GEMM + depthwise.  Test hooks: SHV_CUDA_NO_FUSE=1,
            // SHV_CUDA_FUSE_MASK=<bits> (bit i fuses features[i+1]).
            const char* off = std::getenv("SHV_CUDA_NO_FUSE");
            const char* mask = std::getenv("SHV_CUDA_FUSE_MASK");
            for (int i = 0; i < NBLOCKS; ++i) {
                const BlockHost& b = hm.blocks[i];
                const bool rule = b.cfg.stride == 2 && b.cfg.exp <= 128 && b.cfg.in <= 32;
                const bool want = mask ? ((std::strtoul(mask, nullptr, 0) >> i) & 1ul) != 0 : rule;
                fuse_[i] = !(off && *off) && b.has_expand && want;
                if (fuse_[i]) plan_[i] = plan_expand_dw(b.cfg.in, b.cfg.exp, b.cfg.k, b.cfg.stride, b.h_out, b.w_out);
            }
        }
        upload_weights(hm);
        alloc_buffers(hm);
    }
    ~EngineT() override {
        for (auto& kv : graphs_) cudaGraphExecDestroy(kv.second);
    }

    void encode(int slot, int F, cudaStream_t s) override {
        if (F <= 0) return;
        if (F > cfg_.max_frames) throw std::runtime_error("encode: too many frames");
        if (!cfg_.use_graphs || !dump_dir_.empty()) { run_encode(slot, F, s); return; }
        launch_graph(0, slot, bucket_of(fbuckets_, F), s);
    }
    void classify(int slot, int B, cudaStream_t s) override {
        if (B <= 0) return;
        if (B > cfg_.max_windows) throw std::runtime_error("classify: too many windows");
        if (!cfg_.use_graphs || !dump_dir_.empty()) { run_classify(slot, B, s); return; }
        launch_graph(1, slot, bucket_of(wbuckets_, B), s);
    }

private:
    bool fuse_[NBLOCKS] = {};
    FusedPlan plan_[NBLOCKS];

    // ---------------------------------------------------------------- weights
    struct Blk {
        BlockCfg cfg{};
        bool has_expand = false, residual = false;
        int h_in = 0, w_in = 0, h_out = 0, w_out = 0, cs = 0;
        const T* exp_w = nullptr; const float* exp_b = nullptr;
        const float* dw_w = nullptr; const float* dw_b = nullptr;
        const float *se_w1 = nullptr, *se_b1 = nullptr, *se_w2 = nullptr, *se_b2 = nullptr;
        const T* proj_w = nullptr; const float* proj_b = nullptr;
    };
    struct Gdn {
        const float *norm_w, *ffn_norm_w, *conv_w, *conv_b, *a_log, *dt_bias, *o_norm_w;
        const T *in_w, *o_proj, *ffn_up, *ffn_down;
    };
    DevBuf wbuf_;
    const float *lut_ = nullptr, *stem_w_ = nullptr, *stem_b_ = nullptr;
    Blk blk_[NBLOCKS];
    const T* last_w_ = nullptr; const float* last_b_ = nullptr;
    const T* proj_w_ = nullptr;
    const float *pos_ = nullptr, *readout_ = nullptr;
    Gdn gdn_[2]{};
    const float *s_attn_norm_ = nullptr, *s_b_all_ = nullptr, *s_w_qc_ = nullptr, *s_w_o_ = nullptr, *s_lambda_ = nullptr;
    const float *s_ffn_norm_ = nullptr, *s_ffn_up_ = nullptr, *s_ffn_down_ = nullptr;
    const T* s_w_all_ = nullptr;
    const float *fn_w_ = nullptr, *h_w0t_ = nullptr, *h_b0_ = nullptr, *h_w2_ = nullptr, *h_b2_ = nullptr;

    void upload_weights(const HostModel& hm) {
        Arena a;
        struct Fix { const void** dst; size_t off; };
        std::vector<Fix> fx;
        auto F32 = [&](const float** dst, const std::vector<float>& v) { fx.push_back({reinterpret_cast<const void**>(dst), a.add<float>(v)}); };
        auto TT = [&](const T** dst, const std::vector<float>& v) { fx.push_back({reinterpret_cast<const void**>(dst), a.add<T>(v)}); };
        F32(&lut_, norm_lut());
        F32(&stem_w_, hm.stem_w);
        F32(&stem_b_, hm.stem_b);
        for (int i = 0; i < NBLOCKS; ++i) {
            const BlockHost& b = hm.blocks[i];
            Blk& d = blk_[i];
            d.cfg = b.cfg; d.has_expand = b.has_expand; d.residual = b.residual;
            d.h_in = b.h_in; d.w_in = b.w_in; d.h_out = b.h_out; d.w_out = b.w_out; d.cs = b.se.cs;
            if (b.has_expand) { TT(&d.exp_w, b.expand.w); F32(&d.exp_b, b.expand.b); }
            F32(&d.dw_w, b.dw.w); F32(&d.dw_b, b.dw.b);
            if (b.cfg.se) { F32(&d.se_w1, b.se.w1); F32(&d.se_b1, b.se.b1); F32(&d.se_w2, b.se.w2); F32(&d.se_b2, b.se.b2); }
            TT(&d.proj_w, b.project.w); F32(&d.proj_b, b.project.b);
        }
        TT(&last_w_, hm.last.w); F32(&last_b_, hm.last.b);
        TT(&proj_w_, hm.proj);
        F32(&pos_, hm.pos); F32(&readout_, hm.readout);
        const GdnHost* gh[2] = {&hm.gdn_a, &hm.gdn_b};
        for (int i = 0; i < 2; ++i) {
            Gdn& g = gdn_[i];
            F32(&g.norm_w, gh[i]->norm_w); F32(&g.ffn_norm_w, gh[i]->ffn_norm_w);
            F32(&g.conv_w, gh[i]->conv_w); F32(&g.conv_b, gh[i]->conv_b);
            F32(&g.a_log, gh[i]->a_log); F32(&g.dt_bias, gh[i]->dt_bias); F32(&g.o_norm_w, gh[i]->o_norm_w);
            TT(&g.in_w, gh[i]->in_w); TT(&g.o_proj, gh[i]->o_proj);
            TT(&g.ffn_up, gh[i]->ffn_up); TT(&g.ffn_down, gh[i]->ffn_down);
        }
        const SisaHost& s = hm.sisa;
        F32(&s_attn_norm_, s.attn_norm_w); TT(&s_w_all_, s.w_all); F32(&s_b_all_, s.b_all);
        F32(&s_w_qc_, s.w_qc); F32(&s_w_o_, s.w_o); F32(&s_lambda_, s.lambda_raw);
        F32(&s_ffn_norm_, s.ffn_norm_w); F32(&s_ffn_up_, s.ffn_up); F32(&s_ffn_down_, s.ffn_down);
        F32(&fn_w_, hm.final_norm_w);
        std::vector<float> w0t(size_t(D) * 128);
        for (int i = 0; i < 128; ++i)
            for (int j = 0; j < D; ++j) w0t[size_t(j) * 128 + i] = hm.head_w0[size_t(i) * D + j];
        F32(&h_w0t_, w0t); F32(&h_b0_, hm.head_b0); F32(&h_w2_, hm.head_w2); F32(&h_b2_, hm.head_b2);
        wbuf_.alloc(a.host.size());
        SHV_CUDA_CHECK(cudaMemcpy(wbuf_.p, a.host.data(), a.host.size(), cudaMemcpyHostToDevice));
        for (auto& f : fx) *f.dst = static_cast<const uint8_t*>(wbuf_.p) + f.off;
        weight_bytes_ = a.host.size();
    }

    // ---------------------------------------------------------------- activations
    DevBuf abuf_;
    std::vector<DevBuf> sbufs_;
    T *X_ = nullptr, *E_ = nullptr, *Dd_ = nullptr, *PO_ = nullptr;
    float *se_pool_ = nullptr, *se_hid_ = nullptr, *se_scale_ = nullptr, *se_part_ = nullptr;
    float *Hh_ = nullptr, *P_ = nullptr, *Qs_ = nullptr, *Kn_ = nullptr, *Vc_ = nullptr, *dec_ = nullptr, *beta_ = nullptr, *O_ = nullptr;
    T *XN_ = nullptr, *ON_ = nullptr, *HID_ = nullptr;
    float *XL_ = nullptr, *QC_ = nullptr, *Y_ = nullptr, *X1_ = nullptr, *X2_ = nullptr, *HIDL_ = nullptr;

    void alloc_buffers(const HostModel& hm) {
        const size_t Fm = size_t(cfg_.max_frames), Bm = size_t(cfg_.max_windows), Mm = Bm * NTOK;
        size_t xe = size_t(16) * 90 * 160, ee = size_t(CNN_OUT) * 60, de = 0;
        for (int i = 0; i < NBLOCKS; ++i) {
            const BlockHost& b = hm.blocks[i];
            xe = std::max(xe, size_t(b.cfg.out) * b.h_out * b.w_out);
            if (b.has_expand) ee = std::max(ee, size_t(b.cfg.exp) * b.h_in * b.w_in);
            de = std::max(de, size_t(b.cfg.exp) * b.h_out * b.w_out);
        }
        size_t off = 0;
        std::vector<std::pair<void**, size_t>> plan;
        auto take = [&](void** dst, size_t bytes) { off = (off + 255) & ~size_t(255); plan.push_back({dst, off}); off += bytes; };
        const size_t ts = sizeof(T);
        take(reinterpret_cast<void**>(&X_), Fm * xe * ts);
        take(reinterpret_cast<void**>(&E_), Fm * ee * ts);
        take(reinterpret_cast<void**>(&Dd_), Fm * de * ts);
        take(reinterpret_cast<void**>(&PO_), Fm * 16 * CNN_OUT * ts);
        size_t sep = 0;
        for (int i = 0; i < NBLOCKS; ++i)
            if (hm.blocks[i].cfg.se)
                sep = std::max(sep, size_t(se_splits(hm.blocks[i].h_out * hm.blocks[i].w_out)) * hm.blocks[i].cfg.exp);
        take(reinterpret_cast<void**>(&se_part_), Fm * sep * 4);
        take(reinterpret_cast<void**>(&se_pool_), Fm * CNN_OUT * 4);
        take(reinterpret_cast<void**>(&se_hid_), Fm * 256 * 4);
        take(reinterpret_cast<void**>(&se_scale_), Fm * CNN_OUT * 4);
        take(reinterpret_cast<void**>(&Hh_), Mm * D * 4);
        take(reinterpret_cast<void**>(&P_), Mm * GLD * 4);
        take(reinterpret_cast<void**>(&Qs_), Mm * D * 4);
        take(reinterpret_cast<void**>(&Kn_), Mm * D * 4);
        take(reinterpret_cast<void**>(&Vc_), Mm * VD * 4);
        take(reinterpret_cast<void**>(&dec_), Mm * 8 * 4);
        take(reinterpret_cast<void**>(&beta_), Mm * 8 * 4);
        take(reinterpret_cast<void**>(&O_), Mm * VD * 4);
        take(reinterpret_cast<void**>(&XN_), Mm * D * ts);
        take(reinterpret_cast<void**>(&ON_), Mm * VD * ts);
        take(reinterpret_cast<void**>(&HID_), Mm * DFF * ts);
        take(reinterpret_cast<void**>(&XL_), Bm * D * 4);
        take(reinterpret_cast<void**>(&QC_), Bm * SQN * 4);
        take(reinterpret_cast<void**>(&Y_), Bm * D * 4);
        take(reinterpret_cast<void**>(&X1_), Bm * D * 4);
        take(reinterpret_cast<void**>(&X2_), Bm * D * 4);
        take(reinterpret_cast<void**>(&HIDL_), Bm * DFF * 4);
        abuf_.alloc(off);
        for (auto& p : plan) *p.first = static_cast<uint8_t*>(abuf_.p) + p.second;
        act_bytes_ = off;

        // per-slot input/output buffers
        sbufs_ = std::vector<DevBuf>(size_t(cfg_.slots));
        slots_.resize(size_t(cfg_.slots));
        for (int s = 0; s < cfg_.slots; ++s) {
            size_t so = 0;
            std::vector<std::pair<void**, size_t>> sp;
            auto stake = [&](void** dst, size_t bytes) { so = (so + 255) & ~size_t(255); sp.push_back({dst, so}); so += bytes; };
            Slot& sl = slots_[size_t(s)];
            stake(reinterpret_cast<void**>(&sl.d_frames), Fm * IMG_H * IMG_W * 3);
            stake(reinterpret_cast<void**>(&sl.d_tokens), Fm * 16 * D * 4);
            stake(reinterpret_cast<void**>(&sl.d_tok_stage), Bm * FRAMES * 16 * D * 4);
            stake(reinterpret_cast<void**>(&sl.d_tokptr), Bm * FRAMES * sizeof(float*));
            stake(reinterpret_cast<void**>(&sl.d_logits), Bm * 4);
            sbufs_[size_t(s)].alloc(so);
            for (auto& p : sp) *p.first = static_cast<uint8_t*>(sbufs_[size_t(s)].p) + p.second;
            // every pointer of the table starts valid (padding windows of a bucket read it)
            std::vector<const float*> init(Bm * FRAMES, sl.d_tokens);
            SHV_CUDA_CHECK(cudaMemcpy(sl.d_tokptr, init.data(), init.size() * sizeof(float*), cudaMemcpyHostToDevice));
            slot_bytes_ = so;
        }
    }

    // ---------------------------------------------------------------- graphs
    std::map<long long, cudaGraphExec_t> graphs_;
    void launch_graph(int kind, int slot, int n, cudaStream_t s) {
        const long long key = ((long long)kind << 40) | ((long long)slot << 32) | (long long)n;
        auto it = graphs_.find(key);
        if (it == graphs_.end()) {
            cudaGraph_t g = nullptr;
            SHV_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            try {
                if (kind == 0) run_encode(slot, n, s);
                else run_classify(slot, n, s);
            } catch (...) {
                cudaStreamEndCapture(s, &g);
                if (g) cudaGraphDestroy(g);
                throw;
            }
            SHV_CUDA_CHECK(cudaStreamEndCapture(s, &g));
            cudaGraphExec_t ex = nullptr;
            const cudaError_t e = cudaGraphInstantiate(&ex, g, 0);
            cudaGraphDestroy(g);
            SHV_CUDA_CHECK(e);
            it = graphs_.emplace(key, ex).first;
        }
        SHV_CUDA_CHECK(cudaGraphLaunch(it->second, s));
    }

    // ---------------------------------------------------------------- dumps (debug only)
    template <typename U> void dump(const char* name, const U* dev, size_t n, cudaStream_t s) {
        if (dump_dir_.empty()) return;
        SHV_CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<U> h(n);
        SHV_CUDA_CHECK(cudaMemcpy(h.data(), dev, n * sizeof(U), cudaMemcpyDeviceToHost));
        std::vector<float> f(n);
        for (size_t i = 0; i < n; ++i) f[i] = to_host_float(h[i]);
        std::ofstream o(dump_dir_ + "/" + name + ".f32", std::ios::binary);
        o.write(reinterpret_cast<const char*>(f.data()), std::streamsize(n * sizeof(float)));
    }
    static float to_host_float(float x) { return x; }
    static float to_host_float(__nv_bfloat16 x) { return __bfloat162float(x); }
    static float to_host_float(__half x) { return __half2float(x); }

    static GemmParams gp(int M, int N, int K, const void* A, long long lda, const void* B, long long ldb, void* C,
                         long long ldc, int c_float) {
        GemmParams p;
        p.M = M; p.N = N; p.K = K; p.A = A; p.lda = lda; p.B = B; p.ldb = ldb; p.C = C; p.ldc = ldc; p.c_float = c_float;
        return p;
    }

    // ---------------------------------------------------------------- CNN
    void run_encode(int slot, int F, cudaStream_t s) {
        const Slot& sl = slots_[size_t(slot)];
        launch_stem<T>(sl.d_frames, X_, lut_, stem_w_, stem_b_, F, s);
        SHV_CUDA_CHECK(cudaGetLastError());
        dump("cnn_stem_f0", X_, size_t(16) * 90 * 160, s);
        for (int i = 0; i < NBLOCKS; ++i) {
            const Blk& b = blk_[i];
            const int HWi = b.h_in * b.w_in, HWo = b.h_out * b.w_out;
            const T* dwin = X_;
            bool fused = false;
            if constexpr (!std::is_same<T, float>::value) {
                if (fuse_[i] && dump_dir_.empty()) {
                    launch_expand_dw<T>(plan_[i], X_, b.exp_w, b.exp_b, b.dw_w, b.dw_b, Dd_, F, b.h_in, b.w_in, b.cfg.in,
                                        b.cfg.exp, b.cfg.k, b.cfg.stride, b.h_out, b.w_out, b.cfg.act, s);
                    fused = true;
                }
            }
            if (!fused) {
                if (b.has_expand) {
                    GemmParams p = gp(F * HWi, b.cfg.exp, b.cfg.in, X_, b.cfg.in, b.exp_w, b.cfg.in, E_, b.cfg.exp, 0);
                    p.bias = b.exp_b; p.act = b.cfg.act;
                    gemm<T>(p, s);
                    dwin = E_;
                }
                launch_dw<T>(dwin, Dd_, b.dw_w, b.dw_b, F, b.h_in, b.w_in, b.cfg.exp, b.cfg.k, b.cfg.stride, b.h_out, b.w_out,
                             b.cfg.act, s);
                SHV_CUDA_CHECK(cudaGetLastError());
            }
            const float* scale = nullptr;
            if (b.cfg.se) {
                launch_se_pool<T>(Dd_, se_part_, se_pool_, F, HWo, b.cfg.exp, s);
                SHV_CUDA_CHECK(cudaGetLastError());
                GemmParams p1 = gp(F, b.cs, b.cfg.exp, se_pool_, b.cfg.exp, b.se_w1, b.cfg.exp, se_hid_, b.cs, 1);
                p1.bias = b.se_b1; p1.act = ACT_RELU;
                gemm<float>(p1, s);
                GemmParams p2 = gp(F, b.cfg.exp, b.cs, se_hid_, b.cs, b.se_w2, b.cs, se_scale_, b.cfg.exp, 1);
                p2.bias = b.se_b2; p2.act = ACT_HSIGMOID;
                gemm<float>(p2, s);
                scale = se_scale_;
            }
            GemmParams p = gp(F * HWo, b.cfg.out, b.cfg.exp, Dd_, b.cfg.exp, b.proj_w, b.cfg.exp, X_, b.cfg.out, 0);
            p.bias = b.proj_b;
            if (scale) { p.a_scale = scale; p.scale_rows = HWo; p.ld_scale = b.cfg.exp; }
            if (b.residual) { p.R = X_; p.ldr = b.cfg.out; }
            gemm<T>(p, s);
            if (!dump_dir_.empty()) {
                char name[32];
                std::snprintf(name, sizeof name, "cnn_block%02d_f0", i + 1);
                dump(name, X_, size_t(b.cfg.out) * HWo, s);
                std::snprintf(name, sizeof name, "range_dw%02d_f0", i + 1);
                dump(name, Dd_, size_t(b.cfg.exp) * HWo, s);
                if (b.has_expand) {
                    std::snprintf(name, sizeof name, "range_exp%02d_f0", i + 1);
                    dump(name, E_, size_t(b.cfg.exp) * HWi, s);
                }
            }
        }
        GemmParams pl = gp(F * 60, CNN_OUT, 160, X_, 160, last_w_, 160, E_, CNN_OUT, 0);
        pl.bias = last_b_; pl.act = ACT_HSWISH;
        gemm<T>(pl, s);
        dump("cnn_map_f0", E_, size_t(60) * CNN_OUT, s);
        launch_pool<T>(E_, PO_, F, s);
        SHV_CUDA_CHECK(cudaGetLastError());
        gemm<T>(gp(F * 16, D, CNN_OUT, PO_, CNN_OUT, proj_w_, CNN_OUT, sl.d_tokens, D, 1), s);
        dump("tokens", sl.d_tokens, size_t(F) * 16 * D, s);
    }

    // ---------------------------------------------------------------- sequence
    void run_classify(int slot, int B, cudaStream_t s) {
        const Slot& sl = slots_[size_t(slot)];
        const int M = B * NTOK;
        launch_assemble(sl.d_tokptr, pos_, readout_, Hh_, B, s);
        SHV_CUDA_CHECK(cudaGetLastError());
        dump("h0", Hh_, size_t(NTOK) * D, s);
        for (int st = 0; st < 2; ++st) {
            const Gdn& g = gdn_[st];
            const int reps = st == 0 ? 1 : 2;
            const float rdiv = st == 0 ? 1.0f : float(std::sqrt(2.0));
            for (int r = 0; r < reps; ++r) {
                launch_rmsnorm<T>(Hh_, D, g.norm_w, XN_, D, M, s);
                gemm<T>(gp(M, GN, D, XN_, D, g.in_w, D, P_, GLD, 1), s);
                launch_gdn_prep(P_, g.conv_w, g.conv_b, g.a_log, g.dt_bias, Qs_, Kn_, Vc_, dec_, beta_, B, s);
                launch_delta(Qs_, Kn_, Vc_, dec_, beta_, O_, B, s);
                launch_gated_norm<T>(O_, P_, g.o_norm_w, ON_, M, s);
                SHV_CUDA_CHECK(cudaGetLastError());
                if (!dump_dir_.empty() && st == 0) {
                    dump("gdn_a.q_l2s", Qs_, size_t(NTOK) * D, s);
                    dump("gdn_a.k_l2", Kn_, size_t(NTOK) * D, s);
                    dump("gdn_a.v_conv", Vc_, size_t(NTOK) * VD, s);
                    dump("gdn_a.o_core", O_, size_t(NTOK) * VD, s);
                    dump("gdn_a.o_gated", ON_, size_t(NTOK) * VD, s);
                }
                GemmParams p = gp(M, D, VD, ON_, VD, g.o_proj, VD, Hh_, D, 1);
                p.R = Hh_; p.ldr = D; p.r_float = 1; p.r_div = rdiv;
                gemm<T>(p, s);
            }
            launch_rmsnorm<T>(Hh_, D, g.ffn_norm_w, XN_, D, M, s);
            GemmParams pu = gp(M, 2 * DFF, D, XN_, D, g.ffn_up, D, HID_, DFF, 0);
            pu.swiglu = 1;
            gemm<T>(pu, s);
            GemmParams pd = gp(M, D, DFF, HID_, DFF, g.ffn_down, DFF, Hh_, D, 1);
            pd.R = Hh_; pd.ldr = D; pd.r_float = 1;
            gemm<T>(pd, s);
            dump(st == 0 ? "after_gdn_a" : "after_gdn_b", Hh_, size_t(NTOK) * D, s);
            dump(st == 0 ? "gdn_a.ffn_hidden" : "gdn_b.ffn_hidden", HID_, size_t(NTOK) * DFF, s);
            dump(st == 0 ? "gdn_a.ffn_in" : "gdn_b.ffn_in", XN_, size_t(NTOK) * D, s);
        }
        // SISA: all-row K/V/B/theta/alpha/forget (storage-type GEMM), last-row Q/C and everything after in fp32
        float* S = P_;
        launch_rmsnorm<T>(Hh_, D, s_attn_norm_, XN_, D, M, s);
        GemmParams ps = gp(M, SN, D, XN_, D, s_w_all_, D, S, SLD, 1);
        ps.bias = s_b_all_;
        gemm<T>(ps, s);
        launch_rmsnorm<float>(Hh_ + (NTOK - 1) * D, (long long)NTOK * D, s_attn_norm_, XL_, D, B, s);
        gemm<float>(gp(B, SQN, D, XL_, D, s_w_qc_, D, QC_, SQN, 1), s);
        launch_sisa_attn(S, QC_, s_lambda_, Y_, B, s);
        SHV_CUDA_CHECK(cudaGetLastError());
        GemmParams po = gp(B, D, D, Y_, D, s_w_o_, D, X1_, D, 1);
        po.R = Hh_ + (NTOK - 1) * D; po.ldr = (long long)NTOK * D; po.r_float = 1;
        gemm<float>(po, s);
        launch_rmsnorm<float>(X1_, D, s_ffn_norm_, XL_, D, B, s);
        GemmParams pu = gp(B, 2 * DFF, D, XL_, D, s_ffn_up_, D, HIDL_, DFF, 1);
        pu.swiglu = 1;
        gemm<float>(pu, s);
        GemmParams pd = gp(B, D, DFF, HIDL_, DFF, s_ffn_down_, DFF, X2_, D, 1);
        pd.R = X1_; pd.ldr = D; pd.r_float = 1;
        gemm<float>(pd, s);
        dump("sisa.y", Y_, size_t(D), s);
        dump("after_sisa_row", X2_, size_t(D), s);
        launch_head(X2_, fn_w_, h_w0t_, h_b0_, h_w2_, h_b2_, sl.d_logits, B, s);
        SHV_CUDA_CHECK(cudaGetLastError());
        dump("logit", sl.d_logits, 1, s);
    }
};

}  // namespace

std::unique_ptr<Engine> make_engine(const HostModel& hm, Precision p, const EngineConfig& cfg) {
    switch (p) {
    case Precision::FP32: return std::unique_ptr<Engine>(new EngineT<float>(hm, p, cfg));
    case Precision::BF16: return std::unique_ptr<Engine>(new EngineT<__nv_bfloat16>(hm, p, cfg));
    case Precision::FP16: return std::unique_ptr<Engine>(new EngineT<__half>(hm, p, cfg));
    }
    throw std::runtime_error("unknown precision");
}

}  // namespace shv
