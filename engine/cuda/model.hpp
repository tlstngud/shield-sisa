// SHIELD-V CUDA engine: host-side model description (parsed SHLDV001 file, folded BatchNorm, GEMM layouts).
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace shv {

constexpr int IMG_H = 180, IMG_W = 320, FRAMES = 20, TOK_PER_FRAME = 16, D = 384, NTOK = 321;
constexpr int H = 6, DK = 64, DV = 128, VD = H * DV, DFF = 960, DSTATE = 32, NPAIR = DSTATE / 2;
constexpr int CNN_OUT = 960;
// GDN fused input projection: columns [q 384 | k 384 | v 768 | g 768 | a 6 | b 6] padded to 2320.
constexpr int GQ = 0, GK = 384, GV = 768, GG = 1536, GA = 2304, GB = 2310, GN = 2316, GLD = 2320;
// SISA all-row projection: [K 384 | V 384 | B 192 | theta 96 | alpha 6 | forget 6] padded to 1072.
constexpr int SK = 0, SV = 384, SB = 768, ST_ = 960, SA = 1056, SF = 1062, SN = 1068, SLD = 1072;
// SISA last-row projection: [Q 384 | C 192].
constexpr int SQ = 0, SC = 384, SQN = 576;

// torchvision mobilenet_v3_large inverted-residual settings (features[1..15]); features[16] = 1x1 160->960 HS.
struct BlockCfg { int in, k, exp, out; bool se; int act; int stride; };  // act: 1 ReLU, 2 Hardswish
constexpr int NBLOCKS = 15;
constexpr BlockCfg kBlocks[NBLOCKS] = {
    {16, 3, 16, 16, false, 1, 1},   {16, 3, 64, 24, false, 1, 2},   {24, 3, 72, 24, false, 1, 1},
    {24, 5, 72, 40, true, 1, 2},    {40, 5, 120, 40, true, 1, 1},   {40, 5, 120, 40, true, 1, 1},
    {40, 3, 240, 80, false, 2, 2},  {80, 3, 200, 80, false, 2, 1},  {80, 3, 184, 80, false, 2, 1},
    {80, 3, 184, 80, false, 2, 1},  {80, 3, 480, 112, true, 2, 1},  {112, 3, 672, 112, true, 2, 1},
    {112, 5, 672, 160, true, 2, 2}, {160, 5, 960, 160, true, 2, 1}, {160, 5, 960, 160, true, 2, 1},
};

struct HostTensor { std::vector<int> shape; std::vector<float> data; };

struct PwHost { int in = 0, out = 0; std::vector<float> w, b; };           // w [out][in], b [out]
struct DwHost { int c = 0, k = 0, stride = 1; std::vector<float> w, b; };  // w [k*k][c], b [c]
struct SeHost { int c = 0, cs = 0; std::vector<float> w1, b1, w2, b2; };   // w1 [cs][c], w2 [c][cs]
struct BlockHost {
    BlockCfg cfg{};
    bool has_expand = false, residual = false;
    PwHost expand, project;
    DwHost dw;
    SeHost se;
    int h_in = 0, w_in = 0, h_out = 0, w_out = 0;
};
struct GdnHost {
    std::vector<float> norm_w, ffn_norm_w, in_w /*[GLD][D]*/, conv_w /*[1536][4] q|k|v*/, conv_b /*[1536]*/;
    std::vector<float> a_log, dt_bias, o_norm_w, o_proj /*[D][VD]*/, ffn_up /*[2*DFF][D] interleaved*/, ffn_down /*[D][DFF]*/;
};
struct SisaHost {
    std::vector<float> attn_norm_w, w_all /*[SLD][D]*/, b_all /*[SLD]*/, w_qc /*[SQN][D]*/, w_o /*[D][D]*/;
    std::vector<float> lambda_raw, ffn_norm_w, ffn_up /*[2*DFF][D]*/, ffn_down /*[D][DFF]*/;
};

struct HostModel {
    std::string path, sha256;
    double threshold = 0.0;
    size_t file_bytes = 0;
    std::vector<float> stem_w /*[3][3][3][16] (kh,kw,ci,co)*/, stem_b /*[16]*/;
    BlockHost blocks[NBLOCKS];
    PwHost last;                 // features[16]
    std::vector<float> proj;     // [D][960]
    std::vector<float> pos;      // [320][D] = (time[t] + row[r]) + col[c], token t*16 + r*4 + c
    std::vector<float> readout;  // [D]
    GdnHost gdn_a, gdn_b;
    SisaHost sisa;
    std::vector<float> final_norm_w, head_w0 /*[128][D]*/, head_b0, head_w2 /*[128]*/, head_b2 /*[1]*/;
};

// Parses and validates the SHLDV001 file (every tensor of the architecture must be present with its shape, no
// extra tensors), hashes it, reads the threshold from "<path>.json", folds BatchNorm (eps 1e-3) in double.
HostModel load_host_model(const std::string& path);

}  // namespace shv
