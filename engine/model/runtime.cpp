// SHIELD-V CPU engine runtime (baseline ISA): weight loading and validation, BatchNorm folding, weight packing,
// dispatch, thread pool / teams, workspaces, certificates and the C API (engine/include/shv_cpu.h).
#include "../include/shv_cpu.h"
#include "net.hpp"
#include "sha256.hpp"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef SHV_CPU_REVISION
#define SHV_CPU_REVISION "shvcpu-r1"
#endif
#define SHV_CPU_VERSION_STR "shield_v_cpu 1.0 (" SHV_CPU_REVISION ")"

namespace shv {
namespace {

thread_local std::string g_error;
// per-thread BF16 activation buffer (pairs); GEMMs larger than this are processed in row chunks
constexpr size_t kA16Pairs = 160 * 1024;  // one sequence GEMM at K=960: 321 x 480 = 154,080 pairs

// ------------------------------------------------------------------------------------------------ aligned memory
struct Arena {
    std::vector<void*> blocks;
    size_t bytes = 0;
    template <class T> T* alloc(size_t n) {
        const size_t sz = std::max<size_t>(64, (n * sizeof(T) + 63) / 64 * 64);
        void* p = nullptr;
        if (posix_memalign(&p, 64, sz)) throw std::bad_alloc();
        std::memset(p, 0, sz);
        blocks.push_back(p);
        bytes += sz;
        return static_cast<T*>(p);
    }
    Arena() = default;
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    ~Arena() { for (void* p : blocks) free(p); }
};

// ------------------------------------------------------------------------------------------------ weight file
struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
};
struct TensorMap {
    std::map<std::string, Tensor> t;
    std::set<std::string> used;
    const Tensor& get(const std::string& name, std::initializer_list<int> shape) {
        auto it = t.find(name);
        if (it == t.end()) throw std::runtime_error("weights: missing tensor " + name);
        if (it->second.shape != std::vector<int>(shape)) {
            std::ostringstream o;
            o << "weights: tensor " << name << " has shape [";
            for (size_t i = 0; i < it->second.shape.size(); ++i) o << (i ? "," : "") << it->second.shape[i];
            o << "], expected [";
            size_t i = 0;
            for (int d : shape) o << (i++ ? "," : "") << d;
            o << "]";
            throw std::runtime_error(o.str());
        }
        used.insert(name);
        return it->second;
    }
    const float* f(const std::string& name, std::initializer_list<int> shape) { return get(name, shape).data.data(); }
};

template <class T> T rd(std::ifstream& in) {
    T x{};
    in.read(reinterpret_cast<char*>(&x), sizeof x);
    if (!in) throw std::runtime_error("weights: truncated file");
    return x;
}

void read_weights(const std::string& path, TensorMap& m, std::string& sha) {
    static_assert(sizeof(float) == 4, "float32 required");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open weights: " + path);
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "SHLDV001", 8)) throw std::runtime_error("weights: not a SHLDV001 file: " + path);
    const uint32_t count = rd<uint32_t>(in);
    if (!count || count > 10000) throw std::runtime_error("weights: invalid tensor count");
    uint64_t total = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t n = rd<uint32_t>(in);
        if (!n || n > 4096) throw std::runtime_error("weights: invalid tensor name length");
        std::string name(n, '\0');
        in.read(&name[0], n);
        const uint32_t rank = rd<uint32_t>(in);
        if (rank > 8) throw std::runtime_error("weights: invalid tensor rank");
        Tensor t;
        uint64_t elements = 1;
        for (uint32_t d = 0; d < rank; ++d) {
            const uint32_t dim = rd<uint32_t>(in);
            if (!dim || dim > 10000000 || elements > 100000000 / dim) throw std::runtime_error("weights: invalid shape");
            t.shape.push_back(static_cast<int>(dim));
            elements *= dim;
        }
        if (rd<uint64_t>(in) != elements || (total += elements) > 100000000)
            throw std::runtime_error("weights: invalid tensor size");
        t.data.resize(elements);
        in.read(reinterpret_cast<char*>(t.data.data()), elements * sizeof(float));
        if (!in) throw std::runtime_error("weights: truncated tensor " + name);
        for (float v : t.data)
            if (!std::isfinite(v)) throw std::runtime_error("weights: non-finite value in " + name);
        if (!m.t.emplace(name, std::move(t)).second) throw std::runtime_error("weights: duplicate tensor " + name);
    }
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("weights: trailing data");
    in.clear();
    in.seekg(0);
    sha = sha256_stream(in);
}

// "<weights>.json" sidecar, validated exactly like the CUDA engine: "output_sha256" must be the SHA-256 of the loaded
// weight file (the sidecar belongs to this export), and "threshold" must be present, finite and in (0, 1).
// Any violation fails the load (a missing threshold used to load with NaN, which never alerts).
double read_threshold(const std::string& path, const std::string& sha) {
    const std::string jp = path + ".json";
    std::ifstream in(jp);
    if (!in) throw std::runtime_error("cannot open " + jp + " (threshold)");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    {
        std::string got;
        const size_t k = s.find("\"output_sha256\"");
        const size_t c = k == std::string::npos ? k : s.find(':', k);
        const size_t q0 = c == std::string::npos ? c : s.find('"', c + 1);
        const size_t q1 = q0 == std::string::npos ? q0 : s.find('"', q0 + 1);
        if (q1 != std::string::npos) got = s.substr(q0 + 1, q1 - q0 - 1);
        if (got.empty()) throw std::runtime_error("no output_sha256 in " + jp + " (the sidecar must name its weight file)");
        if (got != sha)
            throw std::runtime_error(jp + " output_sha256 " + got + " does not match the weights' SHA-256 " + sha +
                                     " (sidecar of another export)");
    }
    const size_t k = s.find("\"threshold\"");
    if (k == std::string::npos) throw std::runtime_error("no threshold in " + jp);
    const size_t c = s.find(':', k);
    if (c == std::string::npos) throw std::runtime_error("bad threshold in " + jp);
    const char* b = s.c_str() + c + 1;
    char* e = nullptr;
    const double v = std::strtod(b, &e);
    if (e == b) throw std::runtime_error("bad threshold in " + jp);
    if (!(v > 0.0 && v < 1.0)) throw std::runtime_error("threshold out of (0,1) in " + jp);
    return v;
}

// ------------------------------------------------------------------------------------------------ packing
inline int pad16(int x) { return (x + 15) / 16 * 16; }

// W row-major [N][K] -> 16-column panels; bias [N] or null
Linear pack(Arena& A, const float* W, int N, int K, const float* bias) {
    Linear L{};
    L.K = K; L.N = N; L.Np = pad16(N); L.fmt = FMT_F32;
    float* wp = A.alloc<float>((size_t)L.Np * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) wp[((size_t)(n / 16) * K + k) * 16 + n % 16] = W[(size_t)n * K + k];
    L.wp = wp;
    if (bias) {
        float* b = A.alloc<float>(L.Np);
        std::memcpy(b, bias, sizeof(float) * N);
        L.bias = b;
    }
    return L;
}

constexpr double BN_EPS = 1e-3;  // torchvision MobileNetV3: BatchNorm2d(eps=0.001)
// Fold conv weight [out][per] and BatchNorm (prefix.{weight,bias,running_mean,running_var}) in double.
void fold_bn(TensorMap& m, const float* w, int out, int per, const std::string& bn, std::vector<float>& wf,
             std::vector<float>& bf) {
    const float* g = m.f(bn + ".weight", {out});
    const float* b = m.f(bn + ".bias", {out});
    const float* mu = m.f(bn + ".running_mean", {out});
    const float* var = m.f(bn + ".running_var", {out});
    wf.resize((size_t)out * per);
    bf.resize(out);
    for (int o = 0; o < out; ++o) {
        const double s = (double)g[o] / std::sqrt((double)var[o] + BN_EPS);
        for (int e = 0; e < per; ++e) wf[(size_t)o * per + e] = (float)((double)w[(size_t)o * per + e] * s);
        bf[o] = (float)((double)b[o] - (double)mu[o] * s);
    }
}

int make_divisible(double v, int d = 8) {
    int nv = std::max(d, (int)(v + d / 2.0) / d * d);
    if (nv < 0.9 * v) nv += d;
    return nv;
}

struct BlockCfg { int cin, k, cexp, cout, se, hs, s; };
const BlockCfg kBlocks[15] = {
    {16, 3, 16, 16, 0, 0, 1},    {16, 3, 64, 24, 0, 0, 2},    {24, 3, 72, 24, 0, 0, 1},
    {24, 5, 72, 40, 1, 0, 2},    {40, 5, 120, 40, 1, 0, 1},   {40, 5, 120, 40, 1, 0, 1},
    {40, 3, 240, 80, 0, 1, 2},   {80, 3, 200, 80, 0, 1, 1},   {80, 3, 184, 80, 0, 1, 1},
    {80, 3, 184, 80, 0, 1, 1},   {80, 3, 480, 112, 1, 1, 1},  {112, 3, 672, 112, 1, 1, 1},
    {112, 5, 672, 160, 1, 1, 2}, {160, 5, 960, 160, 1, 1, 1}, {160, 5, 960, 160, 1, 1, 1}};

struct Sizes {  // workspace requirements in floats
    size_t act = 0, dwfull = 0, xfull = 0, ring = 0, strip = 0, zero = 0, stemrows = 3 * 322 * 3, vec = 16 * 960;
};

void build_cnn(TensorMap& m, Arena& A, Cnn& net, Sizes& sz) {
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c)
        for (int v = 0; v < 256; ++v) {
            volatile float x = (float)v / 255.0f;  // x.float().div_(255)
            volatile float y = x - mean[c];         // (x - mean) / std, float32
            net.lut[c][v] = y / stdv[c];
        }
    std::vector<float> wf, bf;
    // stem
    fold_bn(m, m.f("cnn.0.0.weight", {16, 3, 3, 3}), 16, 27, "cnn.0.1", wf, bf);
    float* sw = A.alloc<float>(27 * 16);
    for (int o = 0; o < 16; ++o)
        for (int c = 0; c < 3; ++c)
            for (int dy = 0; dy < 3; ++dy)
                for (int dx = 0; dx < 3; ++dx) sw[((dy * 3 + dx) * 3 + c) * 16 + o] = wf[o * 27 + c * 9 + dy * 3 + dx];
    float* sb = A.alloc<float>(16);
    std::memcpy(sb, bf.data(), sizeof(float) * 16);
    net.stem_w = sw;
    net.stem_b = sb;
    sz.act = 90 * 160 * 16;
    int H = 90, W = 160;
    const char* fs = std::getenv("SHV_CPU_FULL_SOLO_MAX");  // ablation knobs (defaults are the shipped choice)
    const char* ft = std::getenv("SHV_CPU_FULL_TEAM");
    const int full_solo_max = fs && *fs ? std::atoi(fs) : 240;
    const bool full_team = !(ft && std::strcmp(ft, "0") == 0);
    net.nblocks = 15;
    for (int i = 0; i < 15; ++i) {
        const BlockCfg& c = kBlocks[i];
        Block& B = net.blocks[i];
        B = Block{};
        const std::string pre = "cnn." + std::to_string(i + 1) + ".block.";
        B.H = H; B.W = W; B.k = c.k; B.s = c.s; B.p = (c.k - 1) / 2;
        B.Ho = (H + 2 * B.p - c.k) / c.s + 1;
        B.Wo = (W + 2 * B.p - c.k) / c.s + 1;
        B.cin = c.cin; B.cinp = pad16(c.cin); B.cexp = c.cexp; B.cexpp = pad16(c.cexp);
        B.cout = c.cout; B.coutp = pad16(c.cout);
        B.act = c.hs ? ACT_HS : ACT_RELU;
        B.has_expand = c.cexp != c.cin;
        B.has_se = c.se;
        B.residual = c.s == 1 && c.cin == c.cout;
        int j = 0;
        if (B.has_expand) {
            const std::string p = pre + std::to_string(j++);
            fold_bn(m, m.f(p + ".0.weight", {c.cexp, c.cin, 1, 1}), c.cexp, c.cin, p + ".1", wf, bf);
            B.expand = pack(A, wf.data(), c.cexp, c.cin, bf.data());
        }
        {
            const std::string p = pre + std::to_string(j++);
            fold_bn(m, m.f(p + ".0.weight", {c.cexp, 1, c.k, c.k}), c.cexp, c.k * c.k, p + ".1", wf, bf);
            float* dw = A.alloc<float>((size_t)c.k * c.k * B.cexpp);
            for (int ch = 0; ch < c.cexp; ++ch)
                for (int t = 0; t < c.k * c.k; ++t) dw[(size_t)t * B.cexpp + ch] = wf[(size_t)ch * c.k * c.k + t];
            float* db = A.alloc<float>(B.cexpp);
            std::memcpy(db, bf.data(), sizeof(float) * c.cexp);
            B.dw_w = dw;
            B.dw_b = db;
        }
        if (B.has_se) {
            const std::string p = pre + std::to_string(j++);
            B.csq = make_divisible(c.cexp / 4);
            B.csqp = pad16(B.csq);
            const float* w1 = m.f(p + ".fc1.weight", {B.csq, c.cexp, 1, 1});
            B.se1 = pack(A, w1, B.csq, c.cexp, m.f(p + ".fc1.bias", {B.csq}));
            const float* w2 = m.f(p + ".fc2.weight", {c.cexp, B.csq, 1, 1});
            B.se2 = pack(A, w2, c.cexp, B.csq, m.f(p + ".fc2.bias", {c.cexp}));
        }
        {
            const std::string p = pre + std::to_string(j++);
            fold_bn(m, m.f(p + ".0.weight", {c.cout, c.cexp, 1, 1}), c.cout, c.cexp, p + ".1", wf, bf);
            B.project = pack(A, wf.data(), c.cout, c.cexp, bf.data());
        }
        const size_t rw = (size_t)(W + 2 * B.p) * B.cexpp;
        sz.ring = std::max(sz.ring, (size_t)c.k * rw);
        sz.zero = std::max(sz.zero, rw);
        sz.act = std::max(sz.act, (size_t)B.Ho * B.Wo * B.coutp);
        sz.strip = std::max(sz.strip, (size_t)B.Wo * B.cexpp);
        // full-map path: solo for small input maps; in teams for every expanding block except block 2 (the
        // largest map, whose ring overlap is only one row per rank boundary)
        B.full_solo = H * W <= full_solo_max;
        B.full_team = full_team && B.has_expand && i >= 2;
        if (B.full_solo || B.full_team)
            sz.xfull = std::max(sz.xfull, (size_t)(H + 2 * B.p) * (W + 2 * B.p) * B.cexpp);
        if (B.has_se || B.full_solo || B.full_team)
            sz.dwfull = std::max(sz.dwfull, (size_t)B.Ho * B.Wo * B.cexpp);
        H = B.Ho; W = B.Wo;
    }
    if (H != 6 || W != 10) throw std::runtime_error("internal: unexpected CNN map size");
    fold_bn(m, m.f("cnn.16.0.weight", {960, 160, 1, 1}), 960, 160, "cnn.16.1", wf, bf);
    net.last = pack(A, wf.data(), 960, 160, bf.data());
    net.proj = pack(A, m.f("projection.weight", {384, 960}), 384, 960, nullptr);
}

// gate/up of a SwiGLU interleaved per 16-row panel: rows (2i)*16.. = gate[16i..], (2i+1)*16.. = up[16i..]
Linear pack_swiglu(Arena& A, const float* gate, const float* up, int F, int K) {
    std::vector<float> w((size_t)2 * F * K);
    for (int i = 0; i < F / 16; ++i)
        for (int l = 0; l < 16; ++l) {
            std::memcpy(&w[((size_t)(2 * i) * 16 + l) * K], gate + (size_t)(16 * i + l) * K, sizeof(float) * K);
            std::memcpy(&w[((size_t)(2 * i + 1) * 16 + l) * K], up + (size_t)(16 * i + l) * K, sizeof(float) * K);
        }
    return pack(A, w.data(), 2 * F, K, nullptr);
}

float softplus_float(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

void build_seq(TensorMap& m, Arena& A, Seq& S) {
    const int D = SHV_D;
    const float* te = m.f("time_embedding", {20, D});
    const float* re = m.f("row_embedding", {4, D});
    const float* ce = m.f("col_embedding", {4, D});
    float* pos = A.alloc<float>((size_t)320 * D);
    for (int t = 0; t < 20; ++t)
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                for (int d = 0; d < D; ++d) {
                    volatile float a = te[t * D + d] + re[r * D + d];
                    pos[((size_t)t * 16 + r * 4 + c) * D + d] = a + ce[c * D + d];
                }
    S.pos = pos;
    float* ro = A.alloc<float>(D);
    std::memcpy(ro, m.f("readout", {1, 1, D}), sizeof(float) * D);
    S.readout = ro;
    auto vec = [&](const std::string& n, int len) {
        float* p = A.alloc<float>(pad16(len));
        std::memcpy(p, m.f(n, {len}), sizeof(float) * len);
        return (const float*)p;
    };
    for (int g = 0; g < 2; ++g) {
        Gdn& G = S.gdn[g];
        const std::string p = g == 0 ? "gdn_a." : "gdn_b.";
        const std::string x = p + "mixer.";
        G.repeats = g == 0 ? 1 : 2;
        G.norm_w = vec(p + "norm.weight", D);
        G.ffn_norm_w = vec(p + "ffn_norm.weight", D);
        std::vector<float> w((size_t)GDN_N * D);
        auto put = [&](const std::string& n, int rows, int at) {
            std::memcpy(&w[(size_t)at * D], m.f(x + n, {rows, D}), sizeof(float) * rows * D);
        };
        put("q_proj.weight", 384, GDN_Q);
        put("k_proj.weight", 384, GDN_K);
        put("v_proj.weight", 768, GDN_V);
        put("g_proj.weight", 768, GDN_G);
        put("a_proj.weight", 6, GDN_A);
        put("b_proj.weight", 6, GDN_B);
        G.in = pack(A, w.data(), GDN_N, D, nullptr);
        float* cw = A.alloc<float>(4 * 1536);
        float* cb = A.alloc<float>(1536);
        const char* convs[3] = {"q_conv1d", "k_conv1d", "v_conv1d"};
        const int widths[3] = {384, 384, 768}, offs[3] = {0, 384, 768};
        for (int i = 0; i < 3; ++i) {
            const float* wt = m.f(x + convs[i] + ".weight", {widths[i], 1, 4});
            const float* bt = m.f(x + convs[i] + ".bias", {widths[i]});
            for (int c = 0; c < widths[i]; ++c) {
                for (int j = 0; j < 4; ++j) cw[j * 1536 + offs[i] + c] = wt[c * 4 + j];
                cb[offs[i] + c] = bt[c];
            }
        }
        G.conv_w = cw;
        G.conv_b = cb;
        const float* al = m.f(x + "A_log", {6});
        const float* dt = m.f(x + "dt_bias", {6});
        for (int h = 0; h < 6; ++h) {
            G.neg_rate[h] = -expf(al[h]);
            G.dt_bias[h] = dt[h];
        }
        G.o_norm_w = vec(x + "o_norm.weight", 128);
        G.out = pack(A, m.f(x + "o_proj.weight", {384, 768}), 384, 768, nullptr);
        G.gu = pack_swiglu(A, m.f(p + "ffn.gate.weight", {960, D}), m.f(p + "ffn.up.weight", {960, D}), 960, D);
        G.down = pack(A, m.f(p + "ffn.down.weight", {D, 960}), D, 960, nullptr);
    }
    Sisa& Z = S.sisa;
    const std::string a = "sisa.attn.";
    Z.attn_norm_w = vec("sisa.attn_norm.weight", D);
    Z.ffn_norm_w = vec("sisa.ffn_norm.weight", D);
    {
        std::vector<float> w((size_t)SISA_N * D), b(SISA_N, 0.0f);
        auto put = [&](const std::string& n, int rows, int at) {
            std::memcpy(&w[(size_t)at * D], m.f(a + n, {rows, D}), sizeof(float) * rows * D);
        };
        put("W_K.weight", 384, SISA_K);
        put("W_V.weight", 384, SISA_V);
        put("W_B.weight", 192, SISA_B);
        put("theta_proj.weight", 96, SISA_TH);
        put("alpha_proj.weight", 6, SISA_AL);
        put("forget_proj.weight", 6, SISA_FG);
        std::memcpy(&b[SISA_AL], m.f(a + "alpha_proj.bias", {6}), sizeof(float) * 6);
        std::memcpy(&b[SISA_FG], m.f(a + "forget_proj.bias", {6}), sizeof(float) * 6);
        Z.in = pack(A, w.data(), SISA_N, D, b.data());
    }
    Z.q = pack(A, m.f(a + "W_Q.weight", {384, D}), 384, D, nullptr);
    Z.c = pack(A, m.f(a + "W_C.weight", {192, D}), 192, D, nullptr);
    Z.o = pack(A, m.f(a + "W_O.weight", {384, 384}), 384, 384, nullptr);
    const float* lr = m.f(a + "lambda_raw", {6});
    for (int h = 0; h < 6; ++h) Z.lam[h] = softplus_float(lr[h]);
    Z.gu = pack_swiglu(A, m.f("sisa.ffn.w_gate.weight", {960, D}), m.f("sisa.ffn.w_up.weight", {960, D}), 960, D);
    Z.down = pack(A, m.f("sisa.ffn.w_down.weight", {D, 960}), D, 960, nullptr);
    S.final_norm_w = vec("final_norm.weight", D);
    S.head0 = pack(A, m.f("head.0.weight", {128, D}), 128, D, m.f("head.0.bias", {128}));
    {
        float* h2 = A.alloc<float>(128);
        std::memcpy(h2, m.f("head.2.weight", {1, 128}), sizeof(float) * 128);
        S.head2_w = h2;
    }
    S.head2_b = m.f("head.2.bias", {1})[0];
}

// ------------------------------------------------------------------------------------------------ threads
struct alignas(64) Barrier {
    std::atomic<int> count{0};
    std::atomic<int> sense{0};
    int size = 1;
};
void barrier_wait(void* p) {
    Barrier* b = static_cast<Barrier*>(p);
    const int s = b->sense.load(std::memory_order_relaxed);
    if (b->count.fetch_add(1, std::memory_order_acq_rel) == b->size - 1) {
        b->count.store(0, std::memory_order_relaxed);
        b->sense.store(s ^ 1, std::memory_order_release);
        return;
    }
    int spins = 0;
    while (b->sense.load(std::memory_order_acquire) == s) {
        _mm_pause();
        if (++spins > 8192) { std::this_thread::yield(); spins = 0; }
    }
}

class Pool {
public:
    Pool(int n, bool pin) : n_(n) {
        std::vector<int> cpus;
        if (pin) {
            cpu_set_t set;
            CPU_ZERO(&set);
            if (sched_getaffinity(0, sizeof set, &set) == 0)
                for (int c = 0; c < CPU_SETSIZE; ++c)
                    if (CPU_ISSET(c, &set)) cpus.push_back(c);
        }
        for (int i = 1; i < n; ++i) {
            threads_.emplace_back([this, i] { loop(i); });
            if (!cpus.empty()) {
                cpu_set_t one;
                CPU_ZERO(&one);
                CPU_SET(cpus[i % cpus.size()], &one);
                pthread_setaffinity_np(threads_.back().native_handle(), sizeof one, &one);
            }
        }
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> l(mu_);
            stop_ = true;
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    void run(void (*fn)(void*, int), void* ctx) {
        if (n_ == 1) { fn(ctx, 0); return; }
        fn_ = fn;
        ctx_ = ctx;
        remaining_.store(n_ - 1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> l(mu_);
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        fn(ctx, 0);
        int spins = 0;
        while (remaining_.load(std::memory_order_acquire) != 0) {
            if (++spins < 100000) { _mm_pause(); continue; }
            std::unique_lock<std::mutex> l(mu_);
            done_.wait(l, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
            break;
        }
    }

private:
    void loop(int tid) {
        uint64_t seen = 0;
        for (;;) {
            int spins = 0;
            while (gen_.load(std::memory_order_acquire) == seen && spins < 50000) { _mm_pause(); ++spins; }
            if (gen_.load(std::memory_order_acquire) == seen) {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [&] { return gen_.load(std::memory_order_acquire) != seen; });
            }
            seen = gen_.load(std::memory_order_acquire);
            if (stop_) return;
            fn_(ctx_, tid);
            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> l(mu_);
                done_.notify_all();
            }
        }
    }
    int n_;
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_, done_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> remaining_{0};
    void (*fn_)(void*, int) = nullptr;
    void* ctx_ = nullptr;
    bool stop_ = false;
};

}  // namespace

// ------------------------------------------------------------------------------------------------ model
struct Slot {
    FrameBuf fb{};
    CnnScratch cs{};
    SeqBuf sb{};
    SeqScratch ss{};
    std::unique_ptr<Arena> cnn, seq;
};
struct Context {
    std::unique_ptr<Pool> pool;
    std::vector<Slot> slots;
    std::unique_ptr<Barrier[]> barriers;
    std::vector<const uint8_t*> fptr;
    std::vector<float*> tptr;
    std::vector<const float*> cptr;
    std::vector<float, std::allocator<float>> tokbuf;
    size_t bytes() const {
        size_t n = tokbuf.capacity() * sizeof(float);
        for (const Slot& s : slots) n += (s.cnn ? s.cnn->bytes : 0) + (s.seq ? s.seq->bytes : 0);
        return n;
    }
};
}  // namespace shv

struct shv_cpu_model {
    shv::Arena weights;
    shv::Cnn cnn{};
    shv::Seq seq{};
    shv::Sizes sizes;
    const shv::IsaOps* ops = nullptr;
    std::string mode, fingerprint, isa, dump_dir;
    double threshold = 0;
    int threads = 1;
    bool pin = false, bf16 = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::unique_ptr<shv::Context>> all;
    std::vector<shv::Context*> free_list;
};

namespace shv {
namespace {

void ensure_cnn(shv_cpu_model* m, Context* c) {
    const Sizes& z = m->sizes;
    for (Slot& s : c->slots) {
        if (s.cnn) continue;
        s.cnn.reset(new Arena);
        Arena& A = *s.cnn;
        s.fb.act[0] = A.alloc<float>(z.act);
        s.fb.act[1] = A.alloc<float>(z.act);
        s.fb.dwfull = A.alloc<float>(z.dwfull);
        s.fb.xfull = A.alloc<float>(std::max<size_t>(z.xfull, 16));
        s.fb.lastmap = A.alloc<float>(60 * 960);
        s.cs.ring = A.alloc<float>(z.ring);
        s.cs.strip = A.alloc<float>(z.strip);
        s.cs.zero = A.alloc<float>(z.zero);
        s.cs.stemrows = A.alloc<float>(z.stemrows);
        s.cs.vec = A.alloc<float>(z.vec);
        s.cs.ring_floats = z.ring;
        s.cs.strip_floats = z.strip;
        s.cs.strip_rows = 1;
        if (m->bf16) {
            s.cs.a16 = A.alloc<uint32_t>(kA16Pairs);
            s.cs.a16_cap = kA16Pairs;
            s.cs.ascale = A.alloc<float>(4096);
        }
    }
}
void ensure_seq(shv_cpu_model* m, Context* c) {
    for (Slot& s : c->slots) {
        if (s.seq) continue;
        s.seq.reset(new Arena);
        Arena& A = *s.seq;
        s.sb.h = A.alloc<float>((size_t)SHV_T * SHV_D);
        s.sb.hn = A.alloc<float>((size_t)SHV_T * SHV_D);
        s.sb.P = A.alloc<float>((size_t)SHV_T * GDN_P);
        s.sb.O = A.alloc<float>((size_t)SHV_T * SHV_VD);
        s.sb.F = A.alloc<float>((size_t)SHV_T * SHV_FF);
        s.sb.vec = A.alloc<float>(8192);
        s.sb.dvec = A.alloc<double>(1024);
        s.ss.state = A.alloc<float>(64 * 128);
        s.ss.rows = A.alloc<float>(2048);
        if (m->bf16) {
            s.ss.a16 = A.alloc<uint32_t>(kA16Pairs);
            s.ss.a16_cap = kA16Pairs;
        }
    }
}

Context* acquire(shv_cpu_model* m) {
    std::unique_lock<std::mutex> l(m->mu);
    for (;;) {
        if (!m->free_list.empty()) {
            Context* c = m->free_list.back();
            m->free_list.pop_back();
            return c;
        }
        if (m->threads == 1 || m->all.empty()) {
            std::unique_ptr<Context> c(new Context);
            c->pool.reset(new Pool(m->threads, m->pin));
            c->slots.resize(m->threads);
            c->barriers.reset(new Barrier[m->threads]);
            m->all.push_back(std::move(c));
            return m->all.back().get();
        }
        m->cv.wait(l);
    }
}
void release(shv_cpu_model* m, Context* c) {
    {
        std::lock_guard<std::mutex> l(m->mu);
        m->free_list.push_back(c);
    }
    m->cv.notify_one();
}
struct Lease {
    shv_cpu_model* m;
    Context* c;
    explicit Lease(shv_cpu_model* mm) : m(mm), c(acquire(mm)) {}
    ~Lease() { release(m, c); }
};

// Work distribution: items [0, n) over T threads.  Whole items per thread while >= T remain (item i on thread
// i % T), then the remaining r items are processed by teams of T / r threads (team j = threads [j*ts, (j+1)*ts)).
struct Job {
    shv_cpu_model* m;
    Context* c;
    size_t n;
    const uint8_t* const* frames;  // encode
    float* const* tokens;
    const float* const* wtok;      // classify: n*20
    float* logits;
};
template <bool ENC> void run_item(Job* j, size_t item, const Team* tm, Slot& team_slot, Slot& mine) {
    if (ENC) j->m->ops->cnn_frame(&j->m->cnn, j->frames[item], j->tokens[item], tm, &team_slot.fb, &mine.cs);
    else j->m->ops->seq_window(&j->m->seq, j->wtok + item * 20, j->logits + item, tm, &team_slot.sb, &mine.ss);
}
template <bool ENC> void job_fn(void* p, int tid) {
    Job* j = static_cast<Job*>(p);
    const int T = j->m->threads;
    const size_t solo = j->n / T * T;
    const Team one{0, 1, nullptr, nullptr};
    Slot& mine = j->c->slots[tid];
    for (size_t i = tid; i < solo; i += T) run_item<ENC>(j, i, &one, mine, mine);
    const size_t rem = j->n - solo;
    if (!rem) return;
    const int ts = T / (int)rem;
    const int team = tid / ts;
    if (team >= (int)rem) return;
    const Team tm{tid % ts, ts, barrier_wait, &j->c->barriers[team]};
    // the team uses its first member's buffers: wait until every member has finished its solo items
    if (ts > 1) barrier_wait(&j->c->barriers[team]);
    run_item<ENC>(j, solo + team, ts > 1 ? &tm : &one, j->c->slots[team * ts], mine);
}
template <bool ENC> void dispatch(Job& j) {
    const int T = j.m->threads;
    const size_t rem = j.n % T;
    if (rem) {
        const int ts = T / (int)rem;
        for (size_t t = 0; t < rem; ++t) j.c->barriers[t].size = ts;
    }
    if (T == 1) { job_fn<ENC>(&j, 0); return; }
    j.c->pool->run(job_fn<ENC>, &j);
}

void encode(shv_cpu_model* m, Context* c, const uint8_t* const* f, float* const* t, size_t n) {
    ensure_cnn(m, c);
    Job j{m, c, n, f, t, nullptr, nullptr};
    dispatch<true>(j);
}
void classify(shv_cpu_model* m, Context* c, const float* const* wt, size_t b, float* logits) {
    ensure_seq(m, c);
    Job j{m, c, b, nullptr, nullptr, wt, logits};
    dispatch<false>(j);
}

const IsaOps* select_isa(const char* want) {
    __builtin_cpu_init();
    const bool has512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                        __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
                        __builtin_cpu_supports("fma");
    const bool has2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    std::string w = want && *want ? want : "";
    if (w.empty()) {
        const char* e = std::getenv("SHV_CPU_ISA");
        if (e && *e) w = e;
    }
    if (w.empty() || w == "auto") return has512 ? shv_isa_ops_avx512() : has2 ? shv_isa_ops_avx2() : shv_isa_ops_scalar();
    if (w == "avx512") {
        if (!has512) throw std::runtime_error("ISA avx512 not supported by this CPU");
        return shv_isa_ops_avx512();
    }
    if (w == "avx2") {
        if (!has2) throw std::runtime_error("ISA avx2 not supported by this CPU");
        return shv_isa_ops_avx2();
    }
    if (w == "scalar") return shv_isa_ops_scalar();
    throw std::runtime_error("unknown ISA '" + w + "' (avx512, avx2, scalar)");
}

// "<weights>.cpu.cert": line 1 "SHIELD_V_CPU_CERT_V2", then "<weights sha256> <mode> <revision> threshold=<%.17g>"
// per passing mode.  The certified threshold must equal the loaded one exactly (an edited threshold in the sidecar
// invalidates every certified mode).  Returns "" if a matching line exists, else the reason.
std::string cert_check(const std::string& path, const std::string& sha, const std::string& mode, double thr) {
    std::ifstream in(path + ".cpu.cert");
    std::string line;
    if (!std::getline(in, line)) return "no certificate file " + path + ".cpu.cert";
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line != "SHIELD_V_CPU_CERT_V2")
        return "invalid certificate file " + path + ".cpu.cert (expected SHIELD_V_CPU_CERT_V2; V1 lines do not bind the threshold)";
    bool thr_mismatch = false;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string h, n, r, tok;
        if (!(ls >> h >> n >> r)) continue;
        if (!(h == sha && n == mode && r == SHV_CPU_REVISION)) continue;
        double t = std::numeric_limits<double>::quiet_NaN();
        while (ls >> tok)
            if (tok.rfind("threshold=", 0) == 0) {
                const char* b = tok.c_str() + 10;
                char* e = nullptr;
                const double v = std::strtod(b, &e);
                if (e != b && *e == 0) t = v;
            }
        if (std::isfinite(t) && t == thr) return "";
        thr_mismatch = true;
    }
    if (thr_mismatch) {
        char b[96];
        std::snprintf(b, sizeof b, "%.17g", thr);
        return "the certificate for mode '" + mode + "' in " + path + ".cpu.cert does not cover the threshold " + b +
               " in " + path + ".json";
    }
    return "mode '" + mode + "' has no passing certificate in " + path + ".cpu.cert for weights " + sha +
           " (revision " SHV_CPU_REVISION ")";
}

// Modes.  Approximate modes round the inputs and weights of the listed linear layers to BF16 (nearest even) and
// accumulate in FP32 with VDPBF16PS; everything else (stem, depthwise, SE, norms, recurrences, attention, head,
// biases, activations) stays FP32.  They need AVX-512 BF16 and a passing certificate.
// int8-cnn: the CNN 1x1 convolutions with per-row (per pixel) dynamic symmetric INT8 activations and per-output-
// channel INT8 weights, exact int32 accumulation (VPDPBUSD), FP32 dequantisation; int8-cnn-bf16-seq adds bf16-seq.
struct ModeDef { const char* name; bool cnn, seq, cnn8; };
const ModeDef kModes[] = {{"exact", false, false, false},  {"bf16-cnn", true, false, false},
                          {"bf16-seq", false, true, false}, {"bf16", true, true, false},
                          {"int8-cnn", false, false, true}, {"int8-cnn-bf16-seq", false, true, true}};

int round_even(float x) {
    const float lo = std::floor(x), fr = x - lo;
    const int base = (int)lo;
    return base + (int)(fr > 0.5f || (fr == 0.5f && (base & 1)));
}
void to_int8(Arena& A, Linear& L) {
    const int K = L.K, KQ = (K + 3) / 4, np = L.Np / 16;
    int8_t* wq = A.alloc<int8_t>((size_t)np * KQ * 64);
    float* ws = A.alloc<float>(L.Np);
    int32_t* wc = A.alloc<int32_t>(L.Np);
    for (int n = 0; n < L.N; ++n) {
        const float* col = L.wp + (size_t)(n / 16) * K * 16 + n % 16;  // W[n][k] = col[k * 16]
        float mx = 0;
        for (int k = 0; k < K; ++k) mx = std::max(mx, std::fabs(col[(size_t)k * 16]));
        const float s = mx > 0 ? mx / 127.0f : 1.0f;
        ws[n] = s;
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
            const int q = std::min(127, std::max(-127, round_even(col[(size_t)k * 16] / s)));
            wq[(((size_t)(n / 16) * KQ + k / 4) * 16 + n % 16) * 4 + k % 4] = (int8_t)q;
            sum += q;
        }
        wc[n] = 128 * sum;
    }
    L.wq = wq;
    L.wscale = ws;
    L.wcorr = wc;
    L.fmt = FMT_INT8;
}

uint16_t f32_to_bf16(float x) {  // round to nearest even; FP32 subnormals -> signed zero (as VCVTNEPS2BF16)
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) == 0) return uint16_t((u >> 16) & 0x8000u);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return uint16_t((u >> 16) | 0x40u);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
void to_bf16(Arena& A, Linear& L) {
    if (L.K % 2) throw std::runtime_error("internal: BF16 layer with odd K");
    const int KP = L.K / 2, np = L.Np / 16;
    uint16_t* wb = A.alloc<uint16_t>((size_t)np * KP * 32);
    for (int p = 0; p < np; ++p)
        for (int kk = 0; kk < KP; ++kk)
            for (int lane = 0; lane < 16; ++lane)
                for (int e = 0; e < 2; ++e)
                    wb[(((size_t)p * KP + kk) * 16 + lane) * 2 + e] =
                        f32_to_bf16(L.wp[((size_t)p * L.K + 2 * kk + e) * 16 + lane]);
    L.wb = wb;
    L.fmt = FMT_BF16;
}
void apply_mode(Arena& A, Cnn& net, Seq& S, const ModeDef& md) {
    if (md.cnn8) {
        for (int i = 0; i < net.nblocks; ++i) {
            if (net.blocks[i].has_expand) to_int8(A, net.blocks[i].expand);
            to_int8(A, net.blocks[i].project);
        }
        to_int8(A, net.last);
        to_int8(A, net.proj);
    }
    if (md.cnn) {
        for (int i = 0; i < net.nblocks; ++i) {
            if (net.blocks[i].has_expand) to_bf16(A, net.blocks[i].expand);
            to_bf16(A, net.blocks[i].project);
        }
        to_bf16(A, net.last);
        to_bf16(A, net.proj);
    }
    if (md.seq) {
        for (Gdn& g : S.gdn) {
            to_bf16(A, g.in);
            to_bf16(A, g.out);
            to_bf16(A, g.gu);
            to_bf16(A, g.down);
        }
        for (Linear* l : {&S.sisa.in, &S.sisa.q, &S.sisa.c, &S.sisa.o, &S.sisa.gu, &S.sisa.down}) to_bf16(A, *l);
    }
}

// debug hook: raw float32 dump of an intermediate to <SHV_CPU_DUMP_DIR>/<name>.f32 (not used in production)
void dump_file(void* ctx, const char* name, const float* data, size_t n) {
    const std::string path = *static_cast<const std::string*>(ctx) + "/" + name + ".f32";
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
}

shv_cpu_model* load(const char* weights, const char* mode, int threads, const char* isa, bool experimental) {
    try {
        if (!weights || !*weights) throw std::invalid_argument("weights path is empty");
        if (threads < 1 || threads > 64) throw std::invalid_argument("threads must be 1..64");
        std::string md = mode && *mode ? mode : "exact";
        const ModeDef* def = nullptr;
        for (const ModeDef& k : kModes)
            if (md == k.name) def = &k;
        if (!def)
            throw std::invalid_argument("unknown mode '" + md +
                                        "' (exact, bf16-cnn, bf16-seq, bf16, int8-cnn, int8-cnn-bf16-seq)");
        std::unique_ptr<shv_cpu_model> m(new shv_cpu_model);
        m->ops = select_isa(isa);
        m->isa = m->ops->name;
        const bool bf = def->cnn || def->seq || def->cnn8;
        if ((def->cnn || def->seq) && (m->isa != "avx512" || !__builtin_cpu_supports("avx512bf16")))
            throw std::runtime_error("mode '" + md + "' needs the avx512 dispatch level with AVX-512 BF16 (have " +
                                     m->isa + ")");
        if (def->cnn8 && (m->isa != "avx512" || !__builtin_cpu_supports("avx512vnni")))
            throw std::runtime_error("mode '" + md + "' needs the avx512 dispatch level with AVX-512 VNNI (have " +
                                     m->isa + ")");
        TensorMap tm;
        read_weights(weights, tm, m->fingerprint);
        m->threshold = read_threshold(weights, m->fingerprint);
        if (bf && !experimental) {
            const std::string why = cert_check(weights, m->fingerprint, md, m->threshold);
            if (!why.empty()) throw std::runtime_error(why);
        }
        build_cnn(tm, m->weights, m->cnn, m->sizes);
        build_seq(tm, m->weights, m->seq);
        for (const auto& kv : tm.t)
            if (!tm.used.count(kv.first)) throw std::runtime_error("weights: unexpected tensor " + kv.first);
        apply_mode(m->weights, m->cnn, m->seq, *def);
        m->bf16 = bf;
        if (const char* d = std::getenv("SHV_CPU_DUMP_DIR")) {
            if (*d) {
                m->dump_dir = d;
                m->cnn.dump = m->seq.dump = dump_file;
                m->cnn.dump_ctx = m->seq.dump_ctx = &m->dump_dir;
            }
        }
        m->mode = md;
        m->threads = threads;
        const char* pin = std::getenv("SHV_CPU_PIN");
        m->pin = pin && std::strcmp(pin, "1") == 0;
        return m.release();
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return nullptr;
}

template <class F> int guard(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return -1;
}

}  // namespace
}  // namespace shv

using namespace shv;

extern "C" {

shv_cpu_model* shv_cpu_load(const char* w, const char* mode, int threads) { return load(w, mode, threads, nullptr, false); }
shv_cpu_model* shv_cpu_load_ex(const char* w, const char* mode, int threads, const char* isa) {
    return load(w, mode, threads, isa, false);
}
shv_cpu_model* shv_cpu_load_experimental(const char* w, const char* mode, int threads, const char* isa) {
#ifdef SHV_CPU_EXPERIMENTAL
    return load(w, mode, threads, isa, true);
#else
    (void)w; (void)mode; (void)threads; (void)isa;
    g_error = "uncertified modes are disabled in this build (configure with -DSHV_CPU_EXPERIMENTAL=ON)";
    return nullptr;
#endif
}
void shv_cpu_free(shv_cpu_model* m) { delete m; }

int shv_cpu_encode_frame_ptrs(shv_cpu_model* m, const uint8_t* const* frames, size_t n, float* const* tokens) {
    return guard([&] {
        if (!m) throw std::invalid_argument("null model");
        if (!n) return;
        if (!frames || !tokens) throw std::invalid_argument("null frame or token array");
        for (size_t i = 0; i < n; ++i)
            if (!frames[i] || !tokens[i]) throw std::invalid_argument("null frame or token pointer");
        Lease l(m);
        encode(m, l.c, frames, tokens, n);
    });
}
int shv_cpu_encode_frames(shv_cpu_model* m, const uint8_t* rgb, size_t n, float* tokens) {
    return guard([&] {
        if (!m) throw std::invalid_argument("null model");
        if (!n) return;
        if (!rgb || !tokens) throw std::invalid_argument("null input or output");
        Lease l(m);
        Context* c = l.c;
        c->fptr.resize(n);
        c->tptr.resize(n);
        for (size_t i = 0; i < n; ++i) {
            c->fptr[i] = rgb + i * SHV_CPU_FRAME_BYTES;
            c->tptr[i] = tokens + i * SHV_CPU_FRAME_FLOATS;
        }
        encode(m, c, c->fptr.data(), c->tptr.data(), n);
    });
}
int shv_cpu_classify_ptrs(shv_cpu_model* m, const float* const* ft, size_t b, float* logits) {
    return guard([&] {
        if (!m) throw std::invalid_argument("null model");
        if (!b) return;
        if (!ft || !logits) throw std::invalid_argument("null input or output");
        for (size_t i = 0; i < b * 20; ++i)
            if (!ft[i]) throw std::invalid_argument("null frame-token pointer");
        Lease l(m);
        classify(m, l.c, ft, b, logits);
    });
}
int shv_cpu_classify(shv_cpu_model* m, const float* tokens, size_t b, float* logits) {
    return guard([&] {
        if (!m) throw std::invalid_argument("null model");
        if (!b) return;
        if (!tokens || !logits) throw std::invalid_argument("null input or output");
        Lease l(m);
        Context* c = l.c;
        c->cptr.resize(b * 20);
        for (size_t i = 0; i < b * 20; ++i) c->cptr[i] = tokens + i * SHV_CPU_FRAME_FLOATS;
        classify(m, c, c->cptr.data(), b, logits);
    });
}
int shv_cpu_predict(shv_cpu_model* m, const uint8_t* clips, size_t b, float* logits) {
    return guard([&] {
        if (!m) throw std::invalid_argument("null model");
        if (!b) return;
        if (!clips || !logits) throw std::invalid_argument("null input or output");
        Lease l(m);
        Context* c = l.c;
        const size_t n = b * 20;
        if (c->tokbuf.size() < n * SHV_CPU_FRAME_FLOATS) c->tokbuf.resize(n * SHV_CPU_FRAME_FLOATS);
        c->fptr.resize(n);
        c->tptr.resize(n);
        c->cptr.resize(n);
        for (size_t i = 0; i < n; ++i) {
            c->fptr[i] = clips + i * SHV_CPU_FRAME_BYTES;
            c->tptr[i] = c->tokbuf.data() + i * SHV_CPU_FRAME_FLOATS;
            c->cptr[i] = c->tptr[i];
        }
        encode(m, c, c->fptr.data(), c->tptr.data(), n);
        classify(m, c, c->cptr.data(), b, logits);
    });
}
const char* shv_cpu_last_error(void) { return g_error.c_str(); }
const char* shv_cpu_mode(const shv_cpu_model* m) { return m ? m->mode.c_str() : ""; }
const char* shv_cpu_fingerprint(const shv_cpu_model* m) { return m ? m->fingerprint.c_str() : ""; }
const char* shv_cpu_isa(const shv_cpu_model* m) { return m ? m->isa.c_str() : ""; }
int shv_cpu_threads(const shv_cpu_model* m) { return m ? m->threads : 0; }
double shv_cpu_threshold(const shv_cpu_model* m) { return m ? m->threshold : std::nan(""); }
void shv_cpu_memory(const shv_cpu_model* m, size_t* wb, size_t* ws, size_t* slot) {
    if (!m) return;
    if (wb) *wb = m->weights.bytes;
    if (ws) {
        size_t n = 0;
        std::lock_guard<std::mutex> l(const_cast<shv_cpu_model*>(m)->mu);
        for (const auto& c : m->all) n += c->bytes();
        *ws = n;
    }
    if (slot) {
        const Sizes& z = m->sizes;
        const size_t cnn =
            (2 * z.act + z.dwfull + z.xfull + 60 * 960 + z.ring + z.strip + z.zero + z.stemrows + z.vec) * 4;
        const size_t seq = ((size_t)SHV_T * (2 * SHV_D + GDN_P + SHV_VD + SHV_FF) + 8192 + 64 * 128 + 2048) * 4 + 8192;
        *slot = cnn + seq;
    }
}
int shv_cpu_profile_get(const shv_cpu_model* m, double* ns, long* counts, int len) {
    if (!m || !ns || !counts || len <= 0) return 0;
    m->ops->prof_get(ns, counts, len);
#ifdef SHV_CPU_PROFILE
    return std::min(len, 64);
#else
    return 0;
#endif
}
const char* shv_cpu_revision(void) { return SHV_CPU_REVISION; }
const char* shv_cpu_version(void) { return SHV_CPU_VERSION_STR; }

}  // extern "C"
