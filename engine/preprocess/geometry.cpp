// Crop geometry of video_preprocess.py::_boxes (numpy / Python semantics) and frame sampling of
// shield_v/preprocess.py::sample_frames.
#include <algorithm>
#include <cmath>
#include <limits>

#include "internal.hpp"

namespace shvp {

// numpy 2.0 arr_interp (compiled_base.c), left = fp[0], right = fp[-1], xp strictly increasing
static double np_interp(double x, const double* xp, const double* fp, int n) {
    if (n == 1) return (x < xp[0]) ? fp[0] : ((x > xp[0]) ? fp[0] : fp[0]);
    if (x > xp[n - 1]) return fp[n - 1];
    if (x < xp[0]) return fp[0];
    int j = 0;
    while (j + 1 < n && xp[j + 1] <= x) j++;
    if (j == n - 1) return fp[j];
    if (xp[j] == x) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (std::isnan(r)) {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
    }
    return r;
}

// np.median of n doubles (n >= 1)
static double np_median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n & 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

static inline long py_round(double v) { return (long)std::nearbyint(v); }   // Python round(): half to even

void crop_geometry(const shv_face faces[5], shv_rect rects[20], int& anchors) {
    static const int idx[5] = {0, 5, 10, 15, 19};
    double ax[5], cx[5], cy[5];
    std::vector<double> hs;
    int n = 0;
    for (int a = 0; a < 5; a++) {
        if (!faces[a].found) continue;
        const shv_rect& b = faces[a].box;
        ax[n] = (double)idx[a];
        cx[n] = (double)b.x + (double)b.w / 2.0;
        cy[n] = (double)b.y + (double)b.h / 2.0;
        hs.push_back((double)b.h);
        n++;
    }
    anchors = n;
    double xs[20], ys[20];
    long crop_h;
    if (n) {
        for (int i = 0; i < 20; i++) {
            xs[i] = np_interp((double)i, ax, cx, n);
            ys[i] = np_interp((double)i, ax, cy, n);
        }
        crop_h = std::min(180L, std::max(80L, py_round(1.6 * np_median(hs))));
    } else {
        for (int i = 0; i < 20; i++) xs[i] = 160.0, ys[i] = 90.0;
        crop_h = 180;
    }
    long crop_w = std::min(320L, py_round((double)(crop_h * 16) / 9.0));
    if (crop_w == 320) crop_h = 180;
    for (int i = 0; i < 20; i++) {
        long rx = py_round(xs[i] - (double)crop_w / 2.0);
        long ry = py_round(ys[i] - (double)crop_h / 2.0);
        rx = std::min(std::max(rx, 0L), 320L - crop_w);
        ry = std::min(std::max(ry, 0L), 180L - crop_h);
        rects[i] = shv_rect{(int)rx, (int)ry, (int)crop_w, (int)crop_h};
    }
}

bool canvas_info(int w, int h, shv_canvas_info& ci) {
    if (w <= 0 || h <= 0) return false;
    double scale = std::min(720.0 / h, 1280.0 / w);
    ci.src_w = w, ci.src_h = h;
    ci.new_h = (int)std::max(1L, py_round(h * scale));
    ci.new_w = (int)std::max(1L, py_round(w * scale));
    if (ci.new_w > 1280 || ci.new_h > 720) return false;
    ci.left = (1280 - ci.new_w) / 2;
    ci.top = (720 - ci.new_h) / 2;
    ci.mode = (ci.new_h == h && ci.new_w == w) ? 0 : (scale < 1 ? 1 : 2);
    return true;
}

// exact value of a/b (b > 0) rounded to nearest-even double, a, b 128-bit
static double ratio_to_double(__int128 a, __int128 b) {
    bool neg = a < 0;
    unsigned __int128 ua = neg ? (unsigned __int128)(-a) : (unsigned __int128)a, ub = (unsigned __int128)b;
    if (ua == 0) return 0.0;
    const unsigned __int128 lim = (unsigned __int128)1 << 53;
    if (ua < lim && ub < lim) {
        double r = (double)(uint64_t)ua / (double)(uint64_t)ub;   // both exact -> correctly rounded quotient
        return neg ? -r : r;
    }
    auto bitlen = [](unsigned __int128 v) {
        int n = 0;
        while (v) v >>= 1, n++;
        return n;
    };
    int la = bitlen(ua), lb = bitlen(ub);
    int s = std::max(0, 55 + lb - la);              // quotient gets >= 55 significant bits
    unsigned __int128 A = ua << s, q = A / ub, r = A % ub;
    int drop = bitlen(q) - 53;
    unsigned __int128 keep = q >> drop, rem = q & (((unsigned __int128)1 << drop) - 1);
    unsigned __int128 half = (unsigned __int128)1 << (drop - 1);
    if (rem > half || (rem == half && (r != 0 || (keep & 1)))) keep++;
    double res = std::ldexp((double)(uint64_t)keep, drop - s);
    return neg ? -res : res;
}

double pts_seconds(int64_t pts, int64_t num, int64_t den) {
    return ratio_to_double((__int128)pts * num, (__int128)den);
}

}  // namespace shvp

extern "C" {

double shv_pts_seconds(int64_t pts, int64_t num, int64_t den) {
    if (den <= 0) return std::numeric_limits<double>::quiet_NaN();
    return shvp::pts_seconds(pts, num, den);
}

int shv_crop_geometry(const shv_face faces[5], shv_rect rects[20]) {
    if (!faces || !rects) return SHV_EINVAL;
    int a;
    shvp::crop_geometry(faces, rects, a);
    return a;
}

int shv_canvas_info_get(int w, int h, shv_canvas_info* info) {
    if (!info || !shvp::canvas_info(w, h, *info)) {
        shvp::set_error("canvas: bad size");
        return SHV_EINVAL;
    }
    return 0;
}

int shv_sample_window(const int64_t* pts, size_t n, int64_t tb_num, int64_t tb_den, double start,
                      double source_duration, size_t out_index[20], double out_times[20]) {
    using namespace shvp;
    if (!pts || tb_den <= 0 || tb_num <= 0 || !out_index || !out_times) return (set_error("sample: args"), SHV_EINVAL);
    if (!std::isnan(source_duration) && source_duration + 1e-3 < start + 10)
        return (set_error("A complete ten-second source window is required"), SHV_ESHORT);
    if (n == 0) return (set_error("Missing first presentation timestamp"), SHV_ESHORT);
    const double origin = pts_seconds(pts[0], tb_num, tb_den);
    int got = 0;
    for (size_t i = 0; i < n && got < 20; i++) {
        double time = pts_seconds(pts[i], tb_num, tb_den) - origin;
        double target = start + (double)got / 2;
        if (time + 1e-6 < target) continue;
        if (time - target > .25) return (set_error("Frame gap exceeds 250ms"), SHV_EGAP);
        out_index[got] = i;
        out_times[got] = time;
        got++;
    }
    if (got != 20) return (set_error("Need 20 distinct source frames"), SHV_ESHORT);
    for (int i = 0; i < 20; i++)
        for (int j = 0; j < i; j++)
            if (pts[out_index[i]] == pts[out_index[j]]) return (set_error("Need 20 distinct source frames"), SHV_EDUP);
    return 0;
}

void shv_sampler_init(shv_sampler* s, double start) {
    s->start = start;
    s->origin = 0;
    s->next_target = 0;
    s->has_origin = 0;
}

int64_t shv_sampler_push(shv_sampler* s, double t, int64_t* mf, int64_t* mc) {
    if (mf) *mf = 0;
    if (mc) *mc = 0;
    if (!s->has_origin) {
        s->origin = t;
        s->has_origin = 1;
    }
    double rel = t - s->origin;
    int64_t first_missing = -1, nmiss = 0;
    for (;;) {
        double target = s->start + (double)s->next_target / 2;
        if (rel + 1e-6 < target) break;
        if (rel - target > .25) {
            if (first_missing < 0) first_missing = s->next_target;
            nmiss++;
            s->next_target++;
            continue;
        }
        int64_t k = s->next_target++;
        if (mf) *mf = first_missing < 0 ? 0 : first_missing;
        if (mc) *mc = nmiss;
        return k;
    }
    if (mf) *mf = first_missing < 0 ? 0 : first_missing;
    if (mc) *mc = nmiss;
    return -1;
}
}
