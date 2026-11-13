// Host-side SHLDV001 parsing, validation, BatchNorm folding and GEMM weight layouts.
// Compiled with -ffp-contract=off: the position table must equal torch's float32 (time + row) + col bit for bit.
#include "model.hpp"
#include "sha256.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace shv {
namespace {

template <class T> T rd(std::ifstream& in) {
    T x{};
    in.read(reinterpret_cast<char*>(&x), sizeof x);
    if (!in) throw std::runtime_error("truncated weights");
    return x;
}

struct Reader {
    std::unordered_map<std::string, HostTensor> t;
    std::set<std::string> used;
    const HostTensor& get(const std::string& n, std::initializer_list<int> shape) {
        auto it = t.find(n);
        if (it == t.end()) throw std::runtime_error("missing tensor: " + n);
        if (it->second.shape != std::vector<int>(shape)) {
            std::ostringstream s;
            s << "bad shape for " << n << ": [";
            for (int d : it->second.shape) s << d << ",";
            s << "]";
            throw std::runtime_error(s.str());
        }
        used.insert(n);
        return it->second;
    }
    const std::vector<float>& vec(const std::string& n, std::initializer_list<int> shape) { return get(n, shape).data; }
};

double read_threshold(const std::string& json_path) {
    std::ifstream in(json_path);
    if (!in) throw std::runtime_error("cannot open " + json_path + " (threshold)");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    const auto k = s.find("\"threshold\"");
    if (k == std::string::npos) throw std::runtime_error("no threshold in " + json_path);
    const auto c = s.find(':', k);
    if (c == std::string::npos) throw std::runtime_error("bad threshold in " + json_path);
    const double v = std::strtod(s.c_str() + c + 1, nullptr);
    if (!(v > 0.0 && v < 1.0)) throw std::runtime_error("threshold out of (0,1) in " + json_path);
    return v;
}

// Conv (no bias) followed by BatchNorm2d(eps 1e-3), folded in double: w' = w*g/sqrt(v+eps), b' = beta - mean*g/sqrt(v+eps).
void fold_bn(Reader& r, const std::string& bn, int out, std::vector<float>& w, int per_out, std::vector<float>& b) {
    const auto& g = r.vec(bn + ".weight", {out});
    const auto& beta = r.vec(bn + ".bias", {out});
    const auto& mean = r.vec(bn + ".running_mean", {out});
    const auto& var = r.vec(bn + ".running_var", {out});
    b.resize(out);
    for (int o = 0; o < out; ++o) {
        const double s = double(g[o]) / std::sqrt(double(var[o]) + 1e-3);
        for (int i = 0; i < per_out; ++i) w[size_t(o) * per_out + i] = float(double(w[size_t(o) * per_out + i]) * s);
        b[o] = float(double(beta[o]) - double(mean[o]) * s);
    }
}

PwHost pw(Reader& r, const std::string& prefix, int in, int out) {
    PwHost p;
    p.in = in; p.out = out;
    p.w = r.vec(prefix + ".0.weight", {out, in, 1, 1});
    fold_bn(r, prefix + ".1", out, p.w, in, p.b);
    return p;
}

DwHost dw(Reader& r, const std::string& prefix, int c, int k, int stride) {
    std::vector<float> w = r.vec(prefix + ".0.weight", {c, 1, k, k});
    std::vector<float> b;
    fold_bn(r, prefix + ".1", c, w, k * k, b);
    DwHost d;
    d.c = c; d.k = k; d.stride = stride; d.b = b;
    d.w.resize(size_t(k) * k * c);
    for (int ch = 0; ch < c; ++ch)
        for (int t = 0; t < k * k; ++t) d.w[size_t(t) * c + ch] = w[size_t(ch) * k * k + t];
    return d;
}

void copy_rows(std::vector<float>& dst, size_t row0, const std::vector<float>& src, size_t rows, size_t cols) {
    std::memcpy(dst.data() + row0 * cols, src.data(), rows * cols * sizeof(float));
}

std::vector<float> interleave_swiglu(const std::vector<float>& gate, const std::vector<float>& up) {
    std::vector<float> w(size_t(2) * DFF * D);
    for (int j = 0; j < DFF; ++j) {
        std::memcpy(w.data() + size_t(2 * j) * D, gate.data() + size_t(j) * D, D * sizeof(float));
        std::memcpy(w.data() + size_t(2 * j + 1) * D, up.data() + size_t(j) * D, D * sizeof(float));
    }
    return w;
}

GdnHost gdn(Reader& r, const std::string& n) {
    GdnHost g;
    g.norm_w = r.vec(n + ".norm.weight", {D});
    g.ffn_norm_w = r.vec(n + ".ffn_norm.weight", {D});
    const std::string m = n + ".mixer";
    g.in_w.assign(size_t(GLD) * D, 0.0f);
    copy_rows(g.in_w, GQ, r.vec(m + ".q_proj.weight", {H * DK, D}), H * DK, D);
    copy_rows(g.in_w, GK, r.vec(m + ".k_proj.weight", {H * DK, D}), H * DK, D);
    copy_rows(g.in_w, GV, r.vec(m + ".v_proj.weight", {VD, D}), VD, D);
    copy_rows(g.in_w, GG, r.vec(m + ".g_proj.weight", {VD, D}), VD, D);
    copy_rows(g.in_w, GA, r.vec(m + ".a_proj.weight", {H, D}), H, D);
    copy_rows(g.in_w, GB, r.vec(m + ".b_proj.weight", {H, D}), H, D);
    g.conv_w.resize(size_t(1536) * 4);
    g.conv_b.resize(1536);
    const char* convs[3] = {"q_conv1d", "k_conv1d", "v_conv1d"};
    const int widths[3] = {384, 384, 768}, offs[3] = {0, 384, 768};
    for (int i = 0; i < 3; ++i) {
        const auto& w = r.vec(m + "." + convs[i] + ".weight", {widths[i], 1, 4});
        const auto& b = r.vec(m + "." + convs[i] + ".bias", {widths[i]});
        std::memcpy(g.conv_w.data() + size_t(offs[i]) * 4, w.data(), w.size() * sizeof(float));
        std::memcpy(g.conv_b.data() + offs[i], b.data(), b.size() * sizeof(float));
    }
    g.a_log = r.vec(m + ".A_log", {H});
    g.dt_bias = r.vec(m + ".dt_bias", {H});
    g.o_norm_w = r.vec(m + ".o_norm.weight", {DV});
    g.o_proj = r.vec(m + ".o_proj.weight", {D, VD});
    g.ffn_up = interleave_swiglu(r.vec(n + ".ffn.gate.weight", {DFF, D}), r.vec(n + ".ffn.up.weight", {DFF, D}));
    g.ffn_down = r.vec(n + ".ffn.down.weight", {D, DFF});
    return g;
}

}  // namespace

HostModel load_host_model(const std::string& path) {
    HostModel hm;
    hm.path = path;
    Reader r;
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open weights: " + path);
        char magic[8];
        in.read(magic, 8);
        if (!in || std::memcmp(magic, "SHLDV001", 8)) throw std::runtime_error("not a SHLDV001 weight file: " + path);
        const auto count = rd<uint32_t>(in);
        if (!count || count > 10000) throw std::runtime_error("invalid tensor count");
        uint64_t total = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const auto n = rd<uint32_t>(in);
            if (!n || n > 4096) throw std::runtime_error("invalid tensor name length");
            std::string name(n, '\0');
            in.read(&name[0], n);
            const auto rank = rd<uint32_t>(in);
            if (rank > 8) throw std::runtime_error("invalid tensor rank");
            HostTensor t;
            uint64_t elements = 1;
            for (uint32_t d = 0; d < rank; ++d) {
                const auto dim = rd<uint32_t>(in);
                if (!dim || dim > 10000000 || elements > 100000000 / dim) throw std::runtime_error("invalid tensor shape");
                t.shape.push_back(int(dim));
                elements *= dim;
            }
            if (rd<uint64_t>(in) != elements || (total += elements) > 100000000) throw std::runtime_error("invalid tensor size");
            t.data.resize(elements);
            in.read(reinterpret_cast<char*>(t.data.data()), std::streamsize(elements * sizeof(float)));
            if (!in) throw std::runtime_error("truncated tensor: " + name);
            for (float v : t.data)
                if (!std::isfinite(v)) throw std::runtime_error("non-finite value in tensor: " + name);
            if (!r.t.emplace(name, std::move(t)).second) throw std::runtime_error("duplicate tensor: " + name);
        }
        if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing weight data");
        in.clear();
        in.seekg(0, std::ios::end);
        hm.file_bytes = size_t(in.tellg());
        in.seekg(0);
        hm.sha256 = sha256_stream(in);
    }
    hm.threshold = read_threshold(path + ".json");

    // ---------------------------------------------------------------- CNN
    {
        std::vector<float> w = r.vec("cnn.0.0.weight", {16, 3, 3, 3});
        fold_bn(r, "cnn.0.1", 16, w, 27, hm.stem_b);
        hm.stem_w.resize(27 * 16);
        for (int co = 0; co < 16; ++co)
            for (int ci = 0; ci < 3; ++ci)
                for (int kh = 0; kh < 3; ++kh)
                    for (int kw = 0; kw < 3; ++kw)
                        hm.stem_w[((kh * 3 + kw) * 3 + ci) * 16 + co] = w[((co * 3 + ci) * 3 + kh) * 3 + kw];
    }
    int h = 90, wd = 160;
    for (int i = 0; i < NBLOCKS; ++i) {
        const BlockCfg& c = kBlocks[i];
        BlockHost& b = hm.blocks[i];
        b.cfg = c;
        b.has_expand = c.exp != c.in;
        b.residual = c.stride == 1 && c.in == c.out;
        const std::string pre = "cnn." + std::to_string(i + 1) + ".block.";
        int idx = 0;
        if (b.has_expand) b.expand = pw(r, pre + std::to_string(idx++), c.in, c.exp);
        b.dw = dw(r, pre + std::to_string(idx++), c.exp, c.k, c.stride);
        if (c.se) ++idx;  // SE tensors are read below (squeeze width comes from the file)
        b.project = pw(r, pre + std::to_string(idx), c.exp, c.out);
        b.h_in = h; b.w_in = wd;
        const int pad = (c.k - 1) / 2;
        h = (h + 2 * pad - c.k) / c.stride + 1;
        wd = (wd + 2 * pad - c.k) / c.stride + 1;
        b.h_out = h; b.w_out = wd;
    }
    if (h != 6 || wd != 10) throw std::runtime_error("unexpected CNN output size");
    hm.last = pw(r, "cnn.16", 160, CNN_OUT);
    hm.proj = r.vec("projection.weight", {D, CNN_OUT});

    // ---------------------------------------------------------------- positions
    const auto& te = r.vec("time_embedding", {FRAMES, D});
    const auto& re = r.vec("row_embedding", {4, D});
    const auto& ce = r.vec("col_embedding", {4, D});
    hm.pos.resize(size_t(320) * D);
    for (int t = 0; t < FRAMES; ++t)
        for (int rr = 0; rr < 4; ++rr)
            for (int cc = 0; cc < 4; ++cc)
                for (int j = 0; j < D; ++j) {
                    volatile float a = te[size_t(t) * D + j] + re[size_t(rr) * D + j];
                    hm.pos[size_t(t * 16 + rr * 4 + cc) * D + j] = a + ce[size_t(cc) * D + j];
                }
    hm.readout = r.vec("readout", {1, 1, D});

    hm.gdn_a = gdn(r, "gdn_a");
    hm.gdn_b = gdn(r, "gdn_b");

    // ---------------------------------------------------------------- SISA
    {
        SisaHost& s = hm.sisa;
        const std::string a = "sisa.attn.";
        s.attn_norm_w = r.vec("sisa.attn_norm.weight", {D});
        s.w_all.assign(size_t(SLD) * D, 0.0f);
        s.b_all.assign(SLD, 0.0f);
        copy_rows(s.w_all, SK, r.vec(a + "W_K.weight", {D, D}), D, D);
        copy_rows(s.w_all, SV, r.vec(a + "W_V.weight", {D, D}), D, D);
        copy_rows(s.w_all, SB, r.vec(a + "W_B.weight", {H * DSTATE, D}), H * DSTATE, D);
        copy_rows(s.w_all, ST_, r.vec(a + "theta_proj.weight", {H * NPAIR, D}), H * NPAIR, D);
        copy_rows(s.w_all, SA, r.vec(a + "alpha_proj.weight", {H, D}), H, D);
        copy_rows(s.w_all, SF, r.vec(a + "forget_proj.weight", {H, D}), H, D);
        const auto& ab = r.vec(a + "alpha_proj.bias", {H});
        const auto& fb = r.vec(a + "forget_proj.bias", {H});
        for (int i = 0; i < H; ++i) { s.b_all[SA + i] = ab[i]; s.b_all[SF + i] = fb[i]; }
        s.w_qc.assign(size_t(SQN) * D, 0.0f);
        copy_rows(s.w_qc, SQ, r.vec(a + "W_Q.weight", {D, D}), D, D);
        copy_rows(s.w_qc, SC, r.vec(a + "W_C.weight", {H * DSTATE, D}), H * DSTATE, D);
        s.w_o = r.vec(a + "W_O.weight", {D, D});
        s.lambda_raw = r.vec(a + "lambda_raw", {H});
        s.ffn_norm_w = r.vec("sisa.ffn_norm.weight", {D});
        s.ffn_up = interleave_swiglu(r.vec("sisa.ffn.w_gate.weight", {DFF, D}), r.vec("sisa.ffn.w_up.weight", {DFF, D}));
        s.ffn_down = r.vec("sisa.ffn.w_down.weight", {D, DFF});
    }
    hm.final_norm_w = r.vec("final_norm.weight", {D});
    hm.head_w0 = r.vec("head.0.weight", {128, D});
    hm.head_b0 = r.vec("head.0.bias", {128});
    hm.head_w2 = r.vec("head.2.weight", {1, 128});
    hm.head_b2 = r.vec("head.2.bias", {1});

    // SE blocks: squeeze width from the file (torchvision: make_divisible(exp // 4, 8)), shapes cross-checked.
    for (int i = 0; i < NBLOCKS; ++i) {
        BlockHost& b = hm.blocks[i];
        if (!b.cfg.se) continue;
        const std::string se = "cnn." + std::to_string(i + 1) + ".block." + std::to_string(b.has_expand ? 2 : 1);
        auto it = r.t.find(se + ".fc1.weight");
        if (it == r.t.end() || it->second.shape.size() != 4) throw std::runtime_error("missing SE " + se);
        const int cs = it->second.shape[0], c = b.cfg.exp;
        b.se.c = c; b.se.cs = cs;
        b.se.w1 = r.vec(se + ".fc1.weight", {cs, c, 1, 1});
        b.se.b1 = r.vec(se + ".fc1.bias", {cs});
        b.se.w2 = r.vec(se + ".fc2.weight", {c, cs, 1, 1});
        b.se.b2 = r.vec(se + ".fc2.bias", {c});
    }
    for (const auto& kv : r.t)
        if (!r.used.count(kv.first)) throw std::runtime_error("unexpected tensor in weight file: " + kv.first);
    return hm;
}

}  // namespace shv
