// OpenCV 4.13 CascadeClassifier (objdetect/src/cascadedetect.cpp) re-implementation for HAAR stump cascades.
//
// Reproduced semantics (new-format XML):
//   * Data::read: stage.threshold = (float)stageThreshold - 1e-5f; stumps (feature, (float)threshold,
//     (float)leaf0, (float)leaf1); HaarEvaluator::Feature rect weights as float.
//   * detectMultiScaleNoGrouping: factor *= scaleFactor in double; window size cvRound(24*factor) for the
//     stop test, cvRound(24*(float)factor) for min/max filtering; per scale sz = cvRound(W/(float)s),
//     ystep = s >= 2 ? 1 : 2, szi = sz + 1; pyramid image = resize(gray, sz, INTER_LINEAR_EXACT) (copy at s=1);
//     integral sum (int32) and squared sum (int32, wraps -> read as unsigned differences).
//   * nstripes = cvCeil(working_width(scale 0) / 32.), stripeSize = max((H/ystep + n-1)/n, 1)*ystep,
//     rows y < min(n*stripeSize, H) only (so a last row not covered by the stripes is never scanned).
//   * Row scan x += ystep, plus one extra step when the window was rejected at stage 0 (runAt result 0).
//   * setWindow: normrect (1,1,22,22): nf = area*sqsum - sum^2 (double); nf > 0 -> vnf = (float)(1/sqrt(nf)),
//     window valid iff area*vnf < 0.1; else invalid (result -1, no skip).
//   * feature = (w0*S0 + w1*S1 [+ w2*S2]) * vnf in float; stage sum in double, value < thr ? left : right;
//     reject when sum < stage threshold.
//   * candidate Rect(cvRound(x*s), cvRound(y*s), cvRound(24*s), cvRound(24*s)) in float.
//   * groupRectangles(minNeighbors, eps=0.2) with cv::partition (union-find, class labels in order of first
//     member), averages with float 1/n and cvRound, small-inside-large filter; clipObjects to the image.
// Candidate order = (scale, y, x): what OpenCV produces with cv::setNumThreads(1) (body called once with the
// whole stripe range).  All arithmetic orders match the reference; compiled with -ffp-contract=off.
#include <algorithm>
#include <array>
#include <map>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "internal.hpp"
#include "simd.hpp"

extern "C" const unsigned char shv_embedded_cascade[];
extern "C" const size_t shv_embedded_cascade_size;

namespace shvp {

// ---------------------------------------------------------------- minimal XML tree
struct XNode {
    std::string name, text;
    std::vector<std::unique_ptr<XNode>> kids;
    const XNode* child(const char* n) const {
        for (auto& k : kids)
            if (k->name == n) return k.get();
        return nullptr;
    }
};

static bool xml_parse(const char* s, size_t len, XNode& root) {
    std::vector<XNode*> stack{&root};
    size_t i = 0;
    while (i < len) {
        if (s[i] == '<') {
            if (i + 3 < len && !strncmp(s + i, "<!--", 4)) {
                const char* e = strstr(s + i + 4, "-->");
                if (!e) return false;
                i = (size_t)(e - s) + 3;
                continue;
            }
            if (i + 1 < len && (s[i + 1] == '?' || s[i + 1] == '!')) {
                const char* e = (const char*)memchr(s + i, '>', len - i);
                if (!e) return false;
                i = (size_t)(e - s) + 1;
                continue;
            }
            const char* e = (const char*)memchr(s + i, '>', len - i);
            if (!e) return false;
            std::string tag(s + i + 1, e);
            i = (size_t)(e - s) + 1;
            if (!tag.empty() && tag[0] == '/') {
                if (stack.size() < 2) return false;
                std::string nm = tag.substr(1);
                while (!nm.empty() && isspace((unsigned char)nm.back())) nm.pop_back();
                if (stack.back()->name != nm) return false;
                stack.pop_back();
                continue;
            }
            bool selfclose = !tag.empty() && tag.back() == '/';
            if (selfclose) tag.pop_back();
            size_t sp = tag.find_first_of(" \t\r\n");
            std::string nm = tag.substr(0, sp);
            std::unique_ptr<XNode> n(new XNode);
            n->name = nm;
            XNode* raw = n.get();
            stack.back()->kids.push_back(std::move(n));
            if (!selfclose) stack.push_back(raw);
        } else {
            size_t j = i;
            while (j < len && s[j] != '<') j++;
            stack.back()->text.append(s + i, j - i);
            i = j;
        }
    }
    return stack.size() == 1;
}

static bool parse_numbers(const std::string& t, std::vector<double>& v) {
    v.clear();
    const char* p = t.c_str();
    char* e;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        double d = strtod(p, &e);
        if (e == p) return false;
        v.push_back(d);
        p = e;
    }
    return true;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

bool cascade_parse(const char* xml, size_t len, Cascade& c) {
    XNode root;
    if (!xml_parse(xml, len, root)) return (set_error("cascade: XML syntax"), false);
    const XNode* st = root.child("opencv_storage");
    if (!st || st->kids.empty()) return (set_error("cascade: no opencv_storage"), false);
    const XNode* cc = st->child("cascade");
    if (!cc) cc = st->kids[0].get();
    const XNode *stype = cc->child("stageType"), *ftype = cc->child("featureType");
    if (!stype || trim(stype->text) != "BOOST") return (set_error("cascade: only BOOST (new format)"), false);
    if (!ftype || trim(ftype->text) != "HAAR") return (set_error("cascade: only HAAR features"), false);
    const XNode *hn = cc->child("height"), *wn = cc->child("width");
    if (!hn || !wn) return (set_error("cascade: no window size"), false);
    c.win_h = atoi(hn->text.c_str());
    c.win_w = atoi(wn->text.c_str());
    if (c.win_w <= 2 || c.win_h <= 2) return (set_error("cascade: bad window"), false);
    const XNode* fp = cc->child("featureParams");
    if (fp && fp->child("maxCatCount") && atoi(fp->child("maxCatCount")->text.c_str()) != 0)
        return (set_error("cascade: categorical features unsupported"), false);
    const XNode* stages = cc->child("stages");
    const XNode* feats = cc->child("features");
    if (!stages || !feats) return (set_error("cascade: no stages/features"), false);
    static const float THRESHOLD_EPS = 1e-5f;
    std::vector<double> nums;
    for (auto& sn : stages->kids) {
        const XNode* thr = sn->child("stageThreshold");
        const XNode* weak = sn->child("weakClassifiers");
        if (!thr || !weak) return (set_error("cascade: bad stage"), false);
        Cascade::Stage stg;
        stg.threshold = (float)strtod(thr->text.c_str(), nullptr) - THRESHOLD_EPS;
        stg.first = (int)c.stumps.size();
        stg.ntrees = (int)weak->kids.size();
        for (auto& wk : weak->kids) {
            const XNode *in = wk->child("internalNodes"), *lv = wk->child("leafValues");
            if (!in || !lv) return (set_error("cascade: bad weak classifier"), false);
            if (!parse_numbers(in->text, nums) || nums.size() != 4)
                return (set_error("cascade: only stump trees (1 node) are supported"), false);
            Cascade::Stump sp;
            sp.feature = (int)nums[2];
            sp.threshold = (float)nums[3];
            if (!parse_numbers(lv->text, nums) || nums.size() != 2) return (set_error("cascade: bad leaves"), false);
            sp.left = (float)nums[0];
            sp.right = (float)nums[1];
            c.stumps.push_back(sp);
        }
        c.stages.push_back(stg);
    }
    for (auto& fn : feats->kids) {
        Cascade::Feature f{};
        const XNode* rects = fn->child("rects");
        if (!rects) return (set_error("cascade: feature without rects"), false);
        const XNode* tl = fn->child("tilted");
        if (tl && atoi(tl->text.c_str()) != 0) return (set_error("cascade: tilted features unsupported"), false);
        f.nrects = 0;
        for (auto& rn : rects->kids) {
            if (f.nrects >= 3) return (set_error("cascade: >3 rects"), false);
            if (!parse_numbers(rn->text, nums) || nums.size() != 5) return (set_error("cascade: bad rect"), false);
            int r = f.nrects++;
            f.x[r] = (int)nums[0], f.y[r] = (int)nums[1], f.w[r] = (int)nums[2], f.h[r] = (int)nums[3];
            f.weight[r] = (float)nums[4];
        }
        c.features.push_back(f);
    }
    for (auto& sp : c.stumps)
        if (sp.feature < 0 || sp.feature >= (int)c.features.size()) return (set_error("cascade: feature idx"), false);
    if (c.stages.empty()) return (set_error("cascade: no stages"), false);
    return true;
}

// ---------------------------------------------------------------- integral images
void integral_scalar(const uint8_t* img, int w, int h, ptrdiff_t stride, int32_t* sum, uint32_t* sq, int S) {
    for (int x = 0; x <= w; x++) sum[x] = 0, sq[x] = 0;
    for (int y = 0; y < h; y++) {
        const uint8_t* r = img + y * stride;
        int32_t* s1 = sum + (size_t)(y + 1) * S;
        const int32_t* s0 = sum + (size_t)y * S;
        uint32_t* q1 = sq + (size_t)(y + 1) * S;
        const uint32_t* q0 = sq + (size_t)y * S;
        s1[0] = 0, q1[0] = 0;
        int32_t rs = 0;
        uint32_t rq = 0;
        for (int x = 0; x < w; x++) {
            int v = r[x];
            rs += v;
            rq += (uint32_t)(v * v);
            s1[x + 1] = s0[x + 1] + rs;
            q1[x + 1] = q0[x + 1] + rq;
        }
    }
}

// column-deinterleaved variant (see prepare()): column c at (c&1)*H + (c>>1)
static void integral_planar(const uint8_t* img, int w, int h, ptrdiff_t stride, int32_t* sum, uint32_t* sq, int S,
                            int H) {
    memset(sum, 0, sizeof(int32_t) * S);
    memset(sq, 0, sizeof(uint32_t) * S);
    for (int y = 0; y < h; y++) {
        const uint8_t* r = img + y * stride;
        int32_t* s1 = sum + (size_t)(y + 1) * S;
        const int32_t* s0 = sum + (size_t)y * S;
        uint32_t* q1 = sq + (size_t)(y + 1) * S;
        const uint32_t* q0 = sq + (size_t)y * S;
        s1[0] = 0, q1[0] = 0;
        int32_t rs = 0;
        uint32_t rq = 0;
        for (int x = 0; x < w; x += 2) {
            // column x+1 (odd plane, index x/2) and x+2 (even plane, index x/2+1)
            int v = r[x];
            rs += v;
            rq += (uint32_t)(v * v);
            int io = H + (x >> 1);
            s1[io] = s0[io] + rs;
            q1[io] = q0[io] + rq;
            if (x + 1 < w) {
                v = r[x + 1];
                rs += v;
                rq += (uint32_t)(v * v);
                int ie = (x >> 1) + 1;
                s1[ie] = s0[ie] + rs;
                q1[ie] = q0[ie] + rq;
            }
        }
    }
}

// ---------------------------------------------------------------- scalar window evaluation
static inline int csum(const int32_t* p, const int* o) { return p[o[0]] - p[o[1]] - p[o[2]] + p[o[3]]; }
static inline uint32_t csumq(const uint32_t* p, const int* o) { return p[o[0]] - p[o[1]] - p[o[2]] + p[o[3]]; }

static inline bool window_vnf(const HaarPrepared& hp, const int32_t* pw, const uint32_t* pq, float& vnf) {
    int valsum = csum(pw, hp.nofs);
    uint32_t valsq = csumq(pq, hp.nofs);
    const double area = hp.area;
    double nf = area * (double)valsq - (double)valsum * valsum;
    if (nf > 0.) {
        nf = std::sqrt(nf);
        vnf = (float)(1. / nf);
        return area * vnf < 1e-1;
    }
    vnf = 1.f;
    return false;
}

static inline float feat_value(const HaarFeatOfs& f, const int32_t* pw, float vnf) {
    float ret = f.w[0] * (float)csum(pw, f.o[0]) + f.w[1] * (float)csum(pw, f.o[1]);
    if (f.w[2] != 0.0f) ret += f.w[2] * (float)csum(pw, f.o[2]);
    return ret * vnf;
}

// returns true if the stage passes
static inline bool eval_stage(const HaarPrepared& hp, int s, const int32_t* pw, float vnf) {
    double tmp = 0;
    int a = hp.stage_first[s], n = hp.stage_n[s];
    for (int i = a; i < a + n; i++) {
        double value = feat_value(hp.feat[i], pw, vnf);
        tmp += value < hp.thr[i] ? hp.left[i] : hp.right[i];
    }
    return !(tmp < hp.stage_thr[s]);
}

void haar_stage0_scalar(const HaarPrepared& hp, const int32_t* sum, const uint32_t* sq, int y, int npos, int step,
                        int8_t* code, float* vnf) {
    const int S = hp.stride;
    for (int i = 0; i < npos; i++) {
        int x = i * step;
        const int32_t* pw = sum + (size_t)y * S + x;
        const uint32_t* pq = sq + (size_t)y * S + x;
        float v;
        if (!window_vnf(hp, pw, pq, v)) {
            code[i] = -1;
            vnf[i] = v;
            continue;
        }
        vnf[i] = v;
        code[i] = eval_stage(hp, 0, pw, v) ? 1 : 0;
    }
}

// evaluate stages 1.. for the listed windows; returns survivors (written to out_x)
int haar_rest_scalar(const HaarPrepared& hp, const int32_t* sum, int y, const int* xs, const float* vnf, int n,
                     int* out_x) {
    const int S = hp.stride;
    const int ns = (int)hp.stage_n.size();
    int m = 0;
    for (int i = 0; i < n; i++) {
        const int32_t* pw = sum + (size_t)y * S + xs[i];
        bool ok = true;
        for (int s = 1; s < ns && ok; s++) ok = eval_stage(hp, s, pw, vnf[i]);
        if (ok) out_x[m++] = xs[i];
    }
    return m;
}

// half > 0: column-deinterleaved integral layout (column c stored at (c&1)*half + (c>>1)), used by the SIMD
// kernels for the ystep-2 scales so that windows x = 2j, 2j+2, ... are contiguous in every plane.
static void prepare(const Cascade& c, int stride, HaarPrepared& hp, int half = 0) {
    hp.stride = stride;
    auto col = [&](int x) { return half ? (x & 1) * half + (x >> 1) : x; };
    auto ofs4 = [&](int x, int y, int w, int h, int* o) {
        o[0] = col(x) + stride * y;
        o[1] = col(x + w) + stride * y;
        o[2] = col(x) + stride * (y + h);
        o[3] = col(x + w) + stride * (y + h);
    };
    ofs4(1, 1, c.win_w - 2, c.win_h - 2, hp.nofs);
    hp.area = (double)(c.win_w - 2) * (c.win_h - 2);
    hp.feat.resize(c.stumps.size());
    hp.thr.resize(c.stumps.size());
    hp.left.resize(c.stumps.size());
    hp.right.resize(c.stumps.size());
    for (size_t i = 0; i < c.stumps.size(); i++) {
        const auto& sp = c.stumps[i];
        const auto& f = c.features[sp.feature];
        HaarFeatOfs& fo = hp.feat[i];
        fo.nrects = f.nrects;
        for (int r = 0; r < 3; r++) {
            if (r < f.nrects) {
                ofs4(f.x[r], f.y[r], f.w[r], f.h[r], fo.o[r]);
                fo.w[r] = f.weight[r];
            } else {
                ofs4(0, 0, 0, 0, fo.o[r]);
                fo.w[r] = 0.f;
            }
        }
        hp.thr[i] = sp.threshold;
        hp.left[i] = sp.left;
        hp.right[i] = sp.right;
    }
    hp.stage_first.clear();
    hp.stage_n.clear();
    hp.stage_thr.clear();
    for (auto& s : c.stages) {
        hp.stage_first.push_back(s.first);
        hp.stage_n.push_back(s.ntrees);
        hp.stage_thr.push_back(s.threshold);
    }
}

// ---------------------------------------------------------------- grouping (cv::groupRectangles)
static inline bool similar(const shv_rect& r1, const shv_rect& r2, double eps) {
    double delta = eps * (std::min(r1.w, r2.w) + std::min(r1.h, r2.h)) * 0.5;
    return std::abs(r1.x - r2.x) <= delta && std::abs(r1.y - r2.y) <= delta &&
           std::abs(r1.x + r1.w - r2.x - r2.w) <= delta && std::abs(r1.y + r1.h - r2.y - r2.h) <= delta;
}

static int partition(const std::vector<shv_rect>& vec, std::vector<int>& labels, double eps) {
    int N = (int)vec.size();
    std::vector<int> parent(N, -1), rank(N, 0);
    for (int i = 0; i < N; i++) {
        int root = i;
        while (parent[root] >= 0) root = parent[root];
        for (int j = 0; j < N; j++) {
            if (i == j || !similar(vec[i], vec[j], eps)) continue;
            int root2 = j;
            while (parent[root2] >= 0) root2 = parent[root2];
            if (root2 != root) {
                int rk = rank[root], rk2 = rank[root2];
                if (rk > rk2)
                    parent[root2] = root;
                else {
                    parent[root] = root2;
                    rank[root2] += rk == rk2;
                    root = root2;
                }
                int k = j, par;
                while ((par = parent[k]) >= 0) {
                    parent[k] = root;
                    k = par;
                }
                k = i;
                while ((par = parent[k]) >= 0) {
                    parent[k] = root;
                    k = par;
                }
            }
        }
    }
    labels.resize(N);
    int nclasses = 0;
    for (int i = 0; i < N; i++) {
        int root = i;
        while (parent[root] >= 0) root = parent[root];
        if (rank[root] >= 0) rank[root] = ~nclasses++;
        labels[i] = ~rank[root];
    }
    return nclasses;
}

static void group_rectangles(std::vector<shv_rect>& rl, int thr, double eps) {
    if (thr <= 0 || rl.empty()) return;
    std::vector<int> labels;
    int nclasses = partition(rl, labels, eps);
    std::vector<shv_rect> rr(nclasses, shv_rect{0, 0, 0, 0});
    std::vector<int> rw(nclasses, 0);
    for (size_t i = 0; i < labels.size(); i++) {
        int cls = labels[i];
        rr[cls].x += rl[i].x;
        rr[cls].y += rl[i].y;
        rr[cls].w += rl[i].w;
        rr[cls].h += rl[i].h;
        rw[cls]++;
    }
    for (int i = 0; i < nclasses; i++) {
        shv_rect r = rr[i];
        float s = 1.f / rw[i];
        rr[i] = shv_rect{round_even_f(r.x * s), round_even_f(r.y * s), round_even_f(r.w * s), round_even_f(r.h * s)};
    }
    rl.clear();
    for (int i = 0; i < nclasses; i++) {
        shv_rect r1 = rr[i];
        int n1 = rw[i];
        if (n1 <= thr) continue;
        int j;
        for (j = 0; j < nclasses; j++) {
            int n2 = rw[j];
            if (j == i || n2 <= thr) continue;
            shv_rect r2 = rr[j];
            int dx = round_even(r2.w * eps), dy = round_even(r2.h * eps);
            if (i != j && r1.x >= r2.x - dx && r1.y >= r2.y - dy && r1.x + r1.w <= r2.x + r2.w + dx &&
                r1.y + r1.h <= r2.y + r2.h + dy && (n2 > std::max(3, n1) || n1 < 3))
                break;
        }
        if (j == nclasses) rl.push_back(r1);
    }
}

static void clip_objects(int W, int H, std::vector<shv_rect>& v) {
    size_t j = 0;
    for (size_t i = 0; i < v.size(); i++) {
        int x1 = std::max(0, v[i].x), y1 = std::max(0, v[i].y);
        int x2 = std::min(W, v[i].x + v[i].w), y2 = std::min(H, v[i].y + v[i].h);
        if (x2 - x1 > 0 && y2 - y1 > 0) v[j++] = shv_rect{x1, y1, x2 - x1, y2 - y1};
    }
    v.resize(j);
}

// ---------------------------------------------------------------- profiling counters (debug)
static thread_local double g_prof[8];
static thread_local long long g_cnt[4];
// timers only when SHV_PREP_PROFILE is set (they cost ~2% of a detection otherwise)
static const bool g_profile = getenv("SHV_PREP_PROFILE") != nullptr;
static inline double tnow() {
    if (!g_profile) return 0.0;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- detection
struct ScratchHaar {
    std::vector<uint8_t> img;
    std::vector<int32_t> sum;
    std::vector<uint32_t> sq;
    std::vector<int8_t> code;
    std::vector<float> vnf, lvnf;
    std::vector<int> lx, outx;
    std::vector<uint8_t> alive;
    HaarPrepared hp, hp2;
    const Cascade* hp_for = nullptr;
    int hp_stride = -1;
};

int cascade_run(const Cascade& c, const uint8_t* gray, int W, int H, ptrdiff_t gstride, double scaleFactor,
                int min_nb, int minW, int minH, int maxW, int maxH, bool group, std::vector<shv_rect>& out) {
    thread_local ScratchHaar sc;
    out.clear();
    if (!(scaleFactor > 1)) return (set_error("detect: scaleFactor must be > 1"), SHV_EINVAL);
    if (maxH == 0 || maxW == 0) maxW = W, maxH = H;
    if (H < c.win_h || W < c.win_w) return 0;
    std::vector<float> all_scales, scales;
    for (double factor = 1;; factor *= scaleFactor) {
        int ww = round_even(c.win_w * factor), wh = round_even(c.win_h * factor);
        if (ww > W || wh > H) break;
        all_scales.push_back((float)factor);
    }
    for (size_t i = 0; i < all_scales.size(); i++) {
        int ww = round_even_f(c.win_w * all_scales[i]), wh = round_even_f(c.win_h * all_scales[i]);
        if (ww > maxW || wh > maxH) break;
        if (ww < minW || wh < minH) continue;
        scales.push_back(all_scales[i]);
    }
    if (scales.empty() && !all_scales.empty()) {
        std::vector<double> d;
        for (size_t v = 0; v < all_scales.size(); v++) {
            int ww = round_even_f(c.win_w * all_scales[v]), wh = round_even_f(c.win_h * all_scales[v]);
            d.push_back((double)(minW - ww) * (minW - ww) + (double)(minH - wh) * (minH - wh));
        }
        size_t im = 0;
        for (size_t i = 0; i < d.size(); i++)
            if (d[im] > d[i]) im = i;
        scales.push_back(all_scales[im]);
    }
    if (scales.empty()) return 0;

    // scale data
    struct SD { float s; int szw, szh, ystep; };
    std::vector<SD> sd(scales.size());
    for (size_t i = 0; i < scales.size(); i++) {
        float s = scales[i];
        sd[i].s = s;
        sd[i].szw = std::max(round_even_f((float)W / s), 0);
        sd[i].szh = std::max(round_even_f((float)H / s), 0);
        sd[i].ystep = s >= 2 ? 1 : 2;
    }
    int work0 = std::max(sd[0].szw + 1 - c.win_w, 0);
    int nstripes = cv_ceil(work0 / 32.);
    if (nstripes <= 0) return 0;

    // the integral stride: widest level + 1 (scale 0 is the largest)
    int maxw = 0, maxh = 0;
    for (auto& d : sd) maxw = std::max(maxw, d.szw), maxh = std::max(maxh, d.szh);
    const int S = maxw + 1 + 16;   // + padding for vector loads past the right edge
    const int HALF = (maxw + 2) / 2 + 24, S2 = 2 * HALF;
    if (sc.hp_for != &c || sc.hp_stride != S) {
        prepare(c, S, sc.hp);
        prepare(c, S2, sc.hp2, HALF);
        sc.hp_for = &c;
        sc.hp_stride = S;
    }
    sc.img.resize((size_t)maxw * maxh + 64);
    sc.sum.assign((size_t)std::max(S, S2) * (maxh + 1 + 3), 0);
    sc.sq.assign((size_t)std::max(S, S2) * (maxh + 1 + 3), 0);

    haar_stage0_fn st0 = haar_stage0_scalar;
    haar_rest_fn rest = haar_rest_scalar;
    void (*integ)(const uint8_t*, int, int, ptrdiff_t, int32_t*, uint32_t*, int) = integral_scalar;
#if defined(__x86_64__)
    haar_block_fn block = nullptr;
    // AVX-512 level: 16-lane stage 0, but the 8-lane block kernel for later stages (measured faster on Zen 5:
    // 16-position blocks die less coherently); both are exact, so the choice only affects speed.
    if (isa() >= SHV_ISA_AVX512)
        st0 = haar_stage0_avx512, block = getenv("SHV_HAAR_BLOCK16") ? haar_block_avx512 : haar_block_avx2,
        integ = integral_avx512;
    else if (isa() >= SHV_ISA_AVX2) st0 = haar_stage0_avx2, block = haar_block_avx2, integ = integral_avx2;
    (void)rest;
#endif

    struct GrayRows : SrcRows {
        const uint8_t* d;
        ptrdiff_t s;
        const uint8_t* row(int y) override { return d + y * s; }
    } grows;
    grows.d = gray;
    grows.s = gstride;

    std::vector<shv_rect>& cand = out;
    for (size_t si = 0; si < sd.size(); si++) {
        const SD& d = sd[si];
        const uint8_t* img;
        ptrdiff_t istride;
        if (d.szw == W && d.szh == H) {
            img = gray;
            istride = gstride;
        } else {
            thread_local std::map<std::array<int, 4>, std::unique_ptr<CvResizePlan>> plans;
            const std::array<int, 4> pkey{W, H, d.szw, d.szh};
            auto pit = plans.find(pkey);
            if (pit == plans.end()) {
                if (plans.size() > 512) plans.clear();
                std::unique_ptr<CvResizePlan> np(new CvResizePlan);
                if (!cv_resize_plan(W, H, d.szw, d.szh, 1, 5, *np)) return (set_error("detect: pyramid plan"), SHV_EINVAL);
                pit = plans.emplace(pkey, std::move(np)).first;
            }
            const CvResizePlan& plan = *pit->second;
            double ta = tnow();
            cv_resize_region(plan, grows, 0, d.szw, 0, d.szh, sc.img.data(), d.szw);
            g_prof[0] += tnow() - ta;
            img = sc.img.data();
            istride = d.szw;
        }
        // SIMD kernels on ystep-2 scales use the deinterleaved layout and see positions as step 1
        const bool planar = block && d.ystep == 2;
        const HaarPrepared& hp = planar ? sc.hp2 : sc.hp;
        const int kstep = planar ? 1 : d.ystep;
        double tb = tnow();
        if (planar)
            integral_planar(img, d.szw, d.szh, istride, sc.sum.data(), sc.sq.data(), S2, HALF);
        else
            integ(img, d.szw, d.szh, istride, sc.sum.data(), sc.sq.data(), S);
        g_prof[1] += tnow() - tb;
        const int sziw = d.szw + 1, szih = d.szh + 1;
        const int szw = std::max(sziw - c.win_w, 0), szh = std::max(szih - c.win_h, 0);
        const int ystep = d.ystep;
        const int stripe = std::max((szh / ystep + nstripes - 1) / nstripes, 1) * ystep;
        const int y1 = std::min(nstripes * stripe, szh);
        const int winw = round_even_f(c.win_w * d.s), winh = round_even_f(c.win_h * d.s);
        const int npos = (szw + ystep - 1) / ystep;
        if (npos <= 0) continue;
        sc.code.resize(npos + 16);
        sc.vnf.resize(npos + 16);
        sc.lvnf.resize(npos + 16);
        sc.lx.resize(npos + 16);
        sc.outx.resize(npos + 16);
        sc.alive.assign(npos + 32, 0);
        for (int y = 0; y < y1; y += ystep) {
            double tc = tnow();
            st0(hp, sc.sum.data(), sc.sq.data(), y, npos, kstep, sc.code.data(), sc.vnf.data());
            double td = tnow();
            g_prof[2] += td - tc;
            g_cnt[0] += npos;
            // sequential scan with the stage-0 skip
            int nl = 0, m;
            double te;
            if (block) {
                for (int i = 0; i < npos;) {
                    int r = sc.code[i];
                    if (r > 0) sc.alive[i] = 1, nl++;
                    i += (r == 0) ? 2 : 1;
                }
                if (!nl) continue;
                g_cnt[1] += nl;
                te = tnow();
                m = block(hp, sc.sum.data(), y, npos, kstep, sc.alive.data(), sc.vnf.data(), sc.outx.data());
                if (planar)
                    for (int k = 0; k < m; k++) sc.outx[k] *= 2;
                memset(sc.alive.data(), 0, (size_t)npos);
            } else {
                for (int i = 0; i < npos;) {
                    int r = sc.code[i];
                    if (r > 0) {
                        sc.lx[nl] = i * ystep;
                        sc.lvnf[nl] = sc.vnf[i];
                        nl++;
                    }
                    i += (r == 0) ? 2 : 1;
                }
                if (!nl) continue;
                g_cnt[1] += nl;
                te = tnow();
                m = rest(hp, sc.sum.data(), y, sc.lx.data(), sc.lvnf.data(), nl, sc.outx.data());
            }
            g_prof[3] += tnow() - te;
            g_cnt[2] += m;
            for (int k = 0; k < m; k++) {
                int x = sc.outx[k];
                cand.push_back(shv_rect{round_even_f((float)x * d.s), round_even_f((float)y * d.s), winw, winh});
            }
        }
    }
    if (group) group_rectangles(cand, min_nb, 0.2);
    clip_objects(W, H, cand);
    return 0;
}

}  // namespace shvp


extern "C" {

// debug: accumulated per-thread ms {pyramid, integral, stage0, rest} and counts {positions, stage0 survivors
// evaluated further, detections}; reset on read.
__attribute__((visibility("default"))) void shv_prep_debug_haar_profile(double* ms4, long long* cnt3) {
    for (int i = 0; i < 4; i++) ms4[i] = shvp::g_prof[i], shvp::g_prof[i] = 0;
    for (int i = 0; i < 3; i++) cnt3[i] = shvp::g_cnt[i], shvp::g_cnt[i] = 0;
}

shv_cascade* shv_cascade_load(const char* path) {
    using namespace shvp;
    std::string buf;
    if (path) {
        FILE* f = fopen(path, "rb");
        if (!f) return (set_error("cascade: cannot open %s", path), nullptr);
        char tmp[65536];
        size_t n;
        while ((n = fread(tmp, 1, sizeof tmp, f)) > 0) buf.append(tmp, n);
        fclose(f);
    } else {
        buf.assign((const char*)shv_embedded_cascade, shv_embedded_cascade_size);
    }
    std::unique_ptr<shv_cascade> c(new shv_cascade);
    if (!cascade_parse(buf.data(), buf.size(), c->c)) return nullptr;
    return c.release();
}

void shv_cascade_free(shv_cascade* c) { delete c; }

static int detect_common(const shv_cascade* c, const uint8_t* gray, int w, int h, ptrdiff_t stride, double sf,
                         int mn, int min_w, int min_h, int max_w, int max_h, shv_rect* out, int max_out, bool group) {
    using namespace shvp;
    if (!c || !gray || w <= 0 || h <= 0) return (set_error("detect: bad arguments"), SHV_EINVAL);
    std::vector<shv_rect> v;
    int rc = cascade_run(c->c, gray, w, h, stride, sf, mn, min_w, min_h, max_w, max_h, group, v);
    if (rc < 0) return rc;
    for (int i = 0; i < (int)v.size() && i < max_out; i++) out[i] = v[i];
    return (int)v.size();
}

int shv_cascade_detect(const shv_cascade* c, const uint8_t* gray, int w, int h, ptrdiff_t stride, double sf, int mn,
                       int min_w, int min_h, int max_w, int max_h, shv_rect* out, int max_out) {
    return detect_common(c, gray, w, h, stride, sf, mn, min_w, min_h, max_w, max_h, out, max_out, true);
}

int shv_cascade_candidates(const shv_cascade* c, const uint8_t* gray, int w, int h, ptrdiff_t stride, double sf,
                           int min_w, int min_h, int max_w, int max_h, shv_rect* out, int max_out) {
    return detect_common(c, gray, w, h, stride, sf, 0, min_w, min_h, max_w, max_h, out, max_out, false);
}
}
