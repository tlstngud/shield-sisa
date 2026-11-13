// OpenCV 4.13 cv::resize re-implementation for 8-bit images (imgproc/src/resize.cpp), restricted to what the
// SHIELD-V preprocessing uses, but able to produce any rectangular region of the output exactly:
//   INTER_AREA, scale >= 1, non-integer : ResizeArea_Invoker<uchar,float> with computeResizeAreaTab weights
//   INTER_AREA, integer scale           : resizeAreaFast_Invoker (2x2: (sum+2)>>2; else cvRound(sum*(1.f/area)))
//   INTER_LINEAR                        : resizeGeneric_ HResizeLinear/VResizeLinear, 11-bit fixed point
//   INTER_LINEAR_EXACT                  : resize_bitExact<uchar, interpolationLinear> (8.8 fixed point)
//   same size                           : copy
// Exactness notes: the float area path accumulates per output pixel in the reference order
// buf = ((0 + S0*a0) + S1*a1) + ..., sum = (b0*buf0 + b1*buf1) + ..., with separate roundings
// (-ffp-contract=off); zero-weight padding taps (x + 0*S == x for x >= 0) keep that order for SIMD lanes.
#include <algorithm>
#include <cfloat>
#include <cstdlib>

#include "internal.hpp"
#include "simd.hpp"

namespace shvp {

static const int INTER_RESIZE_COEF_BITS = 11;
static const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

int cv_area_tab(int ssize, int dsize, int cn, double scale, std::vector<AreaTab>& tab) {
    tab.clear();
    tab.reserve((size_t)ssize * 2);
    for (int dx = 0; dx < dsize; dx++) {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);
        int sx1 = cv_ceil(fsx1), sx2 = cv_floor(fsx2);
        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);
        if (sx1 - fsx1 > 1e-3) tab.push_back({dx * cn, (sx1 - 1) * cn, (float)((sx1 - fsx1) / cellWidth)});
        for (int sx = sx1; sx < sx2; sx++) tab.push_back({dx * cn, sx * cn, float(1.0 / cellWidth)});
        if (fsx2 - sx2 > 1e-3)
            tab.push_back({dx * cn, sx2 * cn, (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth)});
    }
    return (int)tab.size();
}

static void index_tab(const std::vector<AreaTab>& tab, int dsize, int cn, std::vector<int>& first, int& kmax,
                      std::vector<int>& si, std::vector<float>& al) {
    first.assign(dsize + 1, (int)tab.size());
    for (int k = (int)tab.size() - 1; k >= 0; k--) first[tab[k].di / cn] = k;
    for (int d = dsize - 1; d >= 0; d--)
        if (first[d] > first[d + 1]) first[d] = first[d + 1];
    kmax = 0;
    for (int d = 0; d < dsize; d++) kmax = std::max(kmax, first[d + 1] - first[d]);
    si.assign((size_t)dsize * kmax, 0);
    al.assign((size_t)dsize * kmax, 0.f);
    for (int d = 0; d < dsize; d++) {
        int n = first[d + 1] - first[d];
        for (int j = 0; j < kmax; j++) {
            if (j < n) {
                si[(size_t)d * kmax + j] = tab[first[d] + j].si / cn;
                al[(size_t)d * kmax + j] = tab[first[d] + j].alpha;
            } else {
                // padding tap: same source as the last real tap, weight 0 (adds +0.0f, exact)
                si[(size_t)d * kmax + j] = n ? tab[first[d] + n - 1].si / cn : 0;
                al[(size_t)d * kmax + j] = 0.f;
            }
        }
    }
}

// window-permute tables for the vector horizontal kernels (see area_h_avx2): needs p.eidx [K][ne_pad]
static void build_blocks(CvResizePlan& p, int K) {
    const size_t np = p.ne_pad;
        auto blocks = [&](int B, int win, std::vector<int32_t>& base, std::vector<uint8_t>& ok) {
            size_t nb = np / B;
            base.assign(nb, 0);
            ok.assign(nb, 1);
            for (size_t b = 0; b < nb; b++) {
                int lo = 0x7fffffff, hi = 0;
                for (int l = 0; l < B; l++)
                    for (int j = 0; j < K; j++) {
                        int v = p.eidx[j * np + b * B + l];
                        lo = std::min(lo, v), hi = std::max(hi, v);
                    }
                base[b] = lo;
                ok[b] = (hi - lo) < win;
            }
        };
        blocks(8, 32, p.b8_base, p.b8_ok);
        blocks(16, 64, p.b16_base, p.b16_ok);
        p.b8_dw.assign((size_t)K * np, 0);
        p.b8_sh.assign((size_t)K * np, 0);
        p.b16_idx.assign((size_t)K * np, 0);
        for (size_t e = 0; e < np; e++)
            for (int j = 0; j < K; j++) {
                int r8 = p.eidx[j * np + e] - p.b8_base[e / 8];
                int r16 = p.eidx[j * np + e] - p.b16_base[e / 16];
                if (p.b8_ok[e / 8]) p.b8_dw[j * np + e] = r8 >> 2, p.b8_sh[j * np + e] = (r8 & 3) * 8;
                if (p.b16_ok[e / 16]) p.b16_idx[j * np + e] = (uint8_t)r16;
            }
}

bool cv_resize_plan(int sw, int sh, int dw, int dh, int cn, int interpolation, CvResizePlan& p) {
    p = CvResizePlan();
    p.sw = sw, p.sh = sh, p.dw = dw, p.dh = dh, p.cn = cn;
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || (cn != 1 && cn != 3)) return false;
    if (sw == dw && sh == dh) {
        p.kind = 0;
        return true;
    }
    double inv_scale_x = (double)dw / sw, inv_scale_y = (double)dh / sh;
    double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;
    int iscale_x = round_even(scale_x), iscale_y = round_even(scale_y);
    bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON && std::abs(scale_y - iscale_y) < DBL_EPSILON;

    if (interpolation == 5) {  // INTER_LINEAR_EXACT
        if (is_area_fast && iscale_x == 2 && iscale_y == 2 && cn != 2) {
            interpolation = 3;
        } else {
            p.kind = 4;
            // interpolationLinear<uchar>: scale = 1/inv_scale (softdouble == IEEE double here)
            auto coeffs = [](double inv_scale, int srcsize, int dstsize, std::vector<int>& ofs,
                             std::vector<uint16_t>& cf, int& minofst, int& maxofst) {
                double sc = 1.0 / inv_scale;
                minofst = 0;
                maxofst = dstsize;
                ofs.assign(dstsize, 0);
                cf.assign((size_t)dstsize * 2, 0);
                for (int val = 0; val < dstsize; val++) {
                    double fval = sc * ((double)val + 0.5) - 0.5;
                    int ival = cv_floor(fval);
                    if (ival >= 0 && srcsize > 1) {
                        if (ival < srcsize - 1) {
                            ofs[val] = ival;
                            double fr = fval - (double)ival;
                            int c1 = round_even(fr * 256.0);
                            cf[val * 2 + 1] = (uint16_t)c1;
                            cf[val * 2 + 0] = (uint16_t)(256 - c1);
                        } else {
                            ofs[val] = srcsize - 1;
                            maxofst = std::min(maxofst, val);
                        }
                    } else {
                        minofst = std::max(minofst, val + 1);
                    }
                }
            };
            coeffs(inv_scale_x, sw, dw, p.exofs, p.excoef, p.ex_minx, p.ex_maxx);
            coeffs(inv_scale_y, sh, dh, p.eyofs, p.eycoef, p.ey_miny, p.ey_maxy);
            return true;
        }
    }
    if (interpolation == 1 && is_area_fast && iscale_x == 2 && iscale_y == 2) interpolation = 3;
    if (interpolation == 3) {
        if (!(scale_x >= 1 && scale_y >= 1)) return false;   // area emulated by linear: not needed here
        if (is_area_fast) {
            p.kind = 2;
            p.iscale_x = iscale_x;
            p.iscale_y = iscale_y;
            p.fast_scale = 1.f / (iscale_x * iscale_y);
            return true;
        }
        p.kind = 1;
        cv_area_tab(sw, dw, cn, scale_x, p.xtab);
        cv_area_tab(sh, dh, 1, scale_y, p.ytab);
        index_tab(p.xtab, dw, cn, p.xtab_first, p.xk, p.xsi, p.xal);
        index_tab(p.ytab, dh, 1, p.ytab_first, p.yk, p.ysi, p.yal);
        const size_t ne = (size_t)dw * cn;
        const size_t np = (ne + 15) / 16 * 16;
        p.ne_pad = np;
        p.eidx.assign((size_t)p.xk * np, 0);
        p.ew.assign((size_t)p.xk * np, 0.f);
        for (int dx = 0; dx < dw; dx++)
            for (int c = 0; c < cn; c++)
                for (int j = 0; j < p.xk; j++) {
                    size_t e = (size_t)dx * cn + c;
                    p.eidx[j * np + e] = p.xsi[(size_t)dx * p.xk + j] * cn + c;
                    p.ew[j * np + e] = p.xal[(size_t)dx * p.xk + j];
                }
        // padding elements (e >= ne) point at the last real element's taps with weight 0
        for (size_t e = ne; e < np; e++)
            for (int j = 0; j < p.xk; j++) p.eidx[j * np + e] = p.eidx[j * np + ne - 1];
        build_blocks(p, p.xk);
        return true;
    }
    if (interpolation == 1) {
        p.kind = 3;
        p.lxofs.assign(dw, 0);
        p.lalpha.assign((size_t)dw * 2, 0);
        p.lxmax = dw;
        for (int dx = 0; dx < dw; dx++) {
            float fx = (float)((dx + 0.5) * scale_x - 0.5);
            int sx = cv_floor_f(fx);
            fx -= sx;
            if (sx < 0) fx = 0, sx = 0;
            if (sx + 1 >= sw) {
                p.lxmax = std::min(p.lxmax, dx);
                if (sx >= sw - 1) fx = 0, sx = sw - 1;
            }
            p.lxofs[dx] = sx;
            float c0 = 1.f - fx, c1 = fx;
            int a0 = round_even_f(c0 * INTER_RESIZE_COEF_SCALE), a1 = round_even_f(c1 * INTER_RESIZE_COEF_SCALE);
            p.lalpha[dx * 2] = (int16_t)std::max(-32768, std::min(32767, a0));
            p.lalpha[dx * 2 + 1] = (int16_t)std::max(-32768, std::min(32767, a1));
        }
        p.lyofs.assign(dh, 0);
        p.lbeta.assign((size_t)dh * 2, 0);
        for (int dy = 0; dy < dh; dy++) {
            float fy = (float)((dy + 0.5) * scale_y - 0.5);
            int sy = cv_floor_f(fy);
            fy -= sy;
            p.lyofs[dy] = sy;
            float c0 = 1.f - fy, c1 = fy;
            int b0 = round_even_f(c0 * INTER_RESIZE_COEF_SCALE), b1 = round_even_f(c1 * INTER_RESIZE_COEF_SCALE);
            p.lbeta[dy * 2] = (int16_t)std::max(-32768, std::min(32767, b0));
            p.lbeta[dy * 2 + 1] = (int16_t)std::max(-32768, std::min(32767, b1));
        }
        // element tables: h[e] = S[i0]*w0 + S[i1]*w1 (dx >= xmax: S[sx]*2048, i.e. w = (2048, 0))
        const size_t ne = (size_t)dw * cn, np = (ne + 15) / 16 * 16;
        p.ne_pad = np;
        p.eidx.assign(2 * np, 0);
        p.lw.assign(2 * np, 0);
        for (size_t e = 0; e < np; e++) {
            size_t ee = std::min(e, ne - 1);
            int dx = (int)(ee / cn), c = (int)(ee % cn);
            int sx = p.lxofs[dx] * cn + c;
            bool in = dx < p.lxmax;
            p.eidx[e] = sx;
            p.eidx[np + e] = in ? sx + cn : sx;
            p.lw[e] = e < ne ? (in ? p.lalpha[dx * 2] : INTER_RESIZE_COEF_SCALE) : 0;
            p.lw[np + e] = e < ne ? (in ? p.lalpha[dx * 2 + 1] : 0) : 0;
        }
        build_blocks(p, 2);
        return true;
    }
    return false;
}

static inline int clipi(int x, int a, int b) { return x >= a ? (x < b ? x : b - 1) : a; }

void cv_resize_src_rows(const CvResizePlan& p, int y0, int y1, int& lo, int& hi) {
    switch (p.kind) {
        case 0: lo = y0, hi = y1; return;
        case 1: {
            int a = p.ytab_first[y0], b = p.ytab_first[y1];
            lo = p.sh, hi = 0;
            for (int k = a; k < b; k++) lo = std::min(lo, p.ytab[k].si), hi = std::max(hi, p.ytab[k].si + 1);
            if (lo > hi) lo = hi = 0;
            return;
        }
        case 2: lo = std::min(y0 * p.iscale_y, p.sh), hi = std::min(y1 * p.iscale_y, p.sh); return;
        case 3: {
            lo = clipi(p.lyofs[y0], 0, p.sh);
            hi = clipi(p.lyofs[y1 - 1] + 1, 0, p.sh) + 1;
            return;
        }
        case 4: {
            lo = p.sh, hi = 0;
            for (int dy = y0; dy < y1; dy++) {
                int r0, r1;
                if (dy < p.ey_miny) r0 = r1 = 0;
                else if (dy >= p.ey_maxy) r0 = r1 = p.sh - 1;
                else r0 = p.eyofs[dy], r1 = std::min(p.eyofs[dy] + 1, p.sh - 1);
                lo = std::min(lo, r0), hi = std::max(hi, r1 + 1);
            }
            return;
        }
    }
}

void cv_resize_src_cols(const CvResizePlan& p, int x0, int x1, int& lo, int& hi) {
    switch (p.kind) {
        case 0: lo = x0, hi = x1; return;
        case 1: {
            lo = p.sw, hi = 0;
            for (int d = x0; d < x1; d++)
                for (int k = p.xtab_first[d]; k < p.xtab_first[d + 1]; k++)
                    lo = std::min(lo, p.xtab[k].si / p.cn), hi = std::max(hi, p.xtab[k].si / p.cn + 1);
            if (lo > hi) lo = hi = 0;
            return;
        }
        case 2: lo = std::min(x0 * p.iscale_x, p.sw), hi = std::min(x1 * p.iscale_x, p.sw); return;
        case 3: {
            lo = p.lxofs[x0];
            hi = std::min(p.lxofs[x1 - 1] + 2, p.sw);
            return;
        }
        case 4: {
            lo = p.sw, hi = 0;
            for (int dx = x0; dx < x1; dx++) {
                int c0, c1;
                if (dx < p.ex_minx) c0 = c1 = 0;
                else if (dx >= p.ex_maxx) c0 = c1 = p.exofs[p.dw - 1];
                else c0 = p.exofs[dx], c1 = std::min(p.exofs[dx] + 1, p.sw - 1);
                lo = std::min(lo, c0), hi = std::max(hi, c1 + 1);
            }
            return;
        }
    }
}

// ---------------------------------------------------------------- area (general) kernels
// buf row: horizontal accumulation of one source row for output columns [x0,x1) (cn channels)
typedef void (*area_h_fn)(const CvResizePlan& p, const uint8_t* srow, int x0, int x1, float* buf);
typedef void (*area_v_fn)(float* sum, const float* buf, float beta, int n, bool first);
typedef void (*area_out_fn)(const float* sum, uint8_t* dst, int n);

static void area_h_scalar(const CvResizePlan& p, const uint8_t* S, int x0, int x1, float* buf) {
    const int cn = p.cn;
    for (int dx = x0; dx < x1; dx++) {
        float* b = buf + (size_t)(dx - x0) * cn;
        int a = p.xtab_first[dx], e = p.xtab_first[dx + 1];
        if (cn == 3) {
            float t0 = 0.f, t1 = 0.f, t2 = 0.f;
            for (int k = a; k < e; k++) {
                int sxn = p.xtab[k].si;
                float alpha = p.xtab[k].alpha;
                t0 = t0 + S[sxn] * alpha;
                t1 = t1 + S[sxn + 1] * alpha;
                t2 = t2 + S[sxn + 2] * alpha;
            }
            b[0] = t0, b[1] = t1, b[2] = t2;
        } else {
            float t0 = 0.f;
            for (int k = a; k < e; k++) t0 = t0 + S[p.xtab[k].si] * p.xtab[k].alpha;
            b[0] = t0;
        }
    }
}
static void area_v_scalar(float* sum, const float* buf, float beta, int n, bool first) {
    if (first)
        for (int i = 0; i < n; i++) sum[i] = beta * buf[i];
    else
        for (int i = 0; i < n; i++) sum[i] = sum[i] + beta * buf[i];
}
static void area_out_scalar(const float* sum, uint8_t* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = sat_u8_f(sum[i]);
}

#if defined(__x86_64__)
// AVX2 horizontal pass for cn == 3: 8 output values (channels interleaved) per vector, gathered taps.
// Lane l of a vector covering elements e..e+7 (e = 3*dx + c) uses tap j of pixel dx: source byte
// xsi[dx*xk+j]*3 + c, weight xal[dx*xk+j] (0 for padding taps).
// buf must have 16 floats of slack before and after: whole blocks are written (extra lanes are real values of
// neighbouring output elements, discarded).  Source rows need 64 bytes of readable slack.
SHV_TARGET_AVX2 static void area_h_avx2(const CvResizePlan& p, const uint8_t* S, int x0, int x1, float* buf) {
    const int K = p.xk;
    const size_t np = p.ne_pad;
    const int e0 = x0 * p.cn, e1 = x1 * p.cn;
    float* out = buf - e0;
    const __m256i lo8 = _mm256_set1_epi32(0xff);
    for (int b = e0 / 8; b * 8 < e1; b++) {
        const int e = b * 8;
        __m256 acc = _mm256_setzero_ps();
        if (p.b8_ok[b]) {
            __m256i win = _mm256_loadu_si256((const __m256i*)(S + p.b8_base[b]));
            for (int j = 0; j < K; j++) {
                __m256i dw = _mm256_loadu_si256((const __m256i*)(p.b8_dw.data() + j * np + e));
                __m256i sh = _mm256_loadu_si256((const __m256i*)(p.b8_sh.data() + j * np + e));
                __m256i v = _mm256_and_si256(_mm256_srlv_epi32(_mm256_permutevar8x32_epi32(win, dw), sh), lo8);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_loadu_ps(p.ew.data() + j * np + e)));
            }
        } else {
            for (int j = 0; j < K; j++) {
                __m256i iv = _mm256_loadu_si256((const __m256i*)(p.eidx.data() + j * np + e));
                __m256i g = _mm256_and_si256(_mm256_i32gather_epi32((const int*)S, iv, 1), lo8);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(g), _mm256_loadu_ps(p.ew.data() + j * np + e)));
            }
        }
        _mm256_storeu_ps(out + e, acc);
    }
}

#define SHV_TARGET_VBMI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vbmi,avx2,fma,bmi,bmi2")))
SHV_TARGET_VBMI static void area_h_vbmi(const CvResizePlan& p, const uint8_t* S, int x0, int x1, float* buf) {
    const int K = p.xk;
    const size_t np = p.ne_pad;
    const int e0 = x0 * p.cn, e1 = x1 * p.cn;
    float* out = buf - e0;
    for (int b = e0 / 16; b * 16 < e1; b++) {
        const int e = b * 16;
        __m512 acc = _mm512_setzero_ps();
        if (p.b16_ok[b]) {
            __m512i win = _mm512_loadu_si512((const void*)(S + p.b16_base[b]));
            for (int j = 0; j < K; j++) {
                __m512i ix = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)(p.b16_idx.data() + j * np + e)));
                __m128i v8 = _mm512_castsi512_si128(_mm512_permutexvar_epi8(ix, win));
                __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v8));
                acc = _mm512_add_ps(acc, _mm512_mul_ps(v, _mm512_loadu_ps(p.ew.data() + j * np + e)));
            }
        } else {
            for (int j = 0; j < K; j++) {
                __m512i iv = _mm512_loadu_si512((const void*)(p.eidx.data() + j * np + e));
                __m512i g = _mm512_and_si512(_mm512_i32gather_epi32(iv, (const void*)S, 1), _mm512_set1_epi32(0xff));
                acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_cvtepi32_ps(g), _mm512_loadu_ps(p.ew.data() + j * np + e)));
            }
        }
        _mm512_storeu_ps(out + e, acc);
    }
}
SHV_TARGET_AVX2 static void area_v_avx2(float* sum, const float* buf, float beta, int n, bool first) {
    int i = 0;
    __m256 b = _mm256_set1_ps(beta);
    if (first)
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(sum + i, _mm256_mul_ps(b, _mm256_loadu_ps(buf + i)));
    else
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(sum + i,
                             _mm256_add_ps(_mm256_loadu_ps(sum + i), _mm256_mul_ps(b, _mm256_loadu_ps(buf + i))));
    area_v_scalar(sum + i, buf + i, beta, n - i, first);
}
SHV_TARGET_AVX2 static void area_out_avx2(const float* sum, uint8_t* dst, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a = _mm256_cvtps_epi32(_mm256_loadu_ps(sum + i));        // round half even (MXCSR default)
        __m256i b = _mm256_cvtps_epi32(_mm256_loadu_ps(sum + i + 8));
        __m256i w = _mm256_packs_epi32(a, b);
        w = _mm256_permute4x64_epi64(w, 0xD8);
        __m128i q = _mm_packus_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256(w, 1));
        _mm_storeu_si128((__m128i*)(dst + i), q);
    }
    area_out_scalar(sum + i, dst + i, n - i);
}
#endif

static void area_general_region(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1, uint8_t* dst,
                                ptrdiff_t dstride) {
    const int n = (x1 - x0) * p.cn;
    thread_local std::vector<float> bufv, sumv;
    bufv.resize((size_t)n + 64);
    sumv.resize((size_t)n + 64);
    float* bufp = bufv.data() + 16;   // 16 floats of slack on each side for whole-block kernels
    float* sump = sumv.data();
    area_h_fn hf = area_h_scalar;
    area_v_fn vf = area_v_scalar;
    area_out_fn of = area_out_scalar;
#if defined(__x86_64__)
    if (isa() >= SHV_ISA_AVX2) hf = area_h_avx2, vf = area_v_avx2, of = area_out_avx2;
    if (isa() >= SHV_ISA_AVX512 && cpu_has_vbmi()) hf = area_h_vbmi;
#endif
    int last_sy = -1;
    for (int dy = y0; dy < y1; dy++) {
        int a = p.ytab_first[dy], e = p.ytab_first[dy + 1];
        for (int j = a; j < e; j++) {
            int sy = p.ytab[j].si;
            if (sy != last_sy) {
                hf(p, src.row(sy), x0, x1, bufp);
                last_sy = sy;
            }
            vf(sump, bufp, p.ytab[j].alpha, n, j == a);
        }
        of(sump, dst + (dy - y0) * dstride, n);
    }
}

static void area_fast_region(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1, uint8_t* dst,
                             ptrdiff_t dstride);
#if defined(__x86_64__)
// out[i] = saturate_cast<uchar>((float)s[i] * scale) (cvRound = round half even, as cvtps_epi32)
SHV_TARGET_AVX2 static void cvt_scale_u8_avx2(const int32_t* s, float scale, uint8_t* d, int n) {
    int i = 0;
    const __m256 sc = _mm256_set1_ps(scale);
    for (; i + 16 <= n; i += 16) {
        __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(s + i))), sc));
        __m256i b = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(s + i + 8))), sc));
        __m256i w = _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xD8);
        _mm_storeu_si128((__m128i*)(d + i), _mm_packus_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256(w, 1)));
    }
    for (; i < n; i++) d[i] = sat_u8_f((float)s[i] * scale);
}
SHV_TARGET_AVX2 static void vsum_u16_avx2(const uint8_t* const* rows, int nr, int lo, int hi, uint16_t* rs) {
    int i = lo;
    for (; i + 16 <= hi; i += 16) {
        __m256i a = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(rows[0] + i)));
        for (int r = 1; r < nr; r++) a = _mm256_add_epi16(a, _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(rows[r] + i))));
        _mm256_storeu_si256((__m256i*)(rs + i), a);
    }
    for (; i < hi; i++) {
        int v = 0;
        for (int r = 0; r < nr; r++) v += rows[r][i];
        rs[i] = (uint16_t)v;
    }
}
#endif

// Integer-scale area: full blocks of full rows via column sums (exact integers); the rest via the reference's
// count path.  Same arithmetic as resizeAreaFast_Invoker / ResizeAreaFastVec.
static bool area_fast_region_vec(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1, uint8_t* dst,
                                 ptrdiff_t dstride) {
#if defined(__x86_64__)
    if (isa() < SHV_ISA_AVX2) return false;
    const int cn = p.cn, ix = p.iscale_x, iy = p.iscale_y;
    if (ix * iy > 256 || ix * iy <= 4) return false;   // uint16 column sums; 2x2 is faster in the direct scalar loop (measured)
    const int dwidth1 = p.sw / ix;
    const bool fast2 = ix == 2 && iy == 2 && (cn == 1 || cn == 3);
    const int xf1 = std::min(x1, dwidth1);   // full-block columns [x0, xf1)
    if (xf1 <= x0) return false;
    thread_local std::vector<uint16_t> rs;
    thread_local std::vector<int32_t> sums;
    rs.resize((size_t)p.sw * cn + 32);
    sums.resize((size_t)(x1 - x0) * cn + 32);
    std::vector<const uint8_t*> rows(iy);
    for (int dy = y0; dy < y1; dy++) {
        int sy0 = dy * iy;
        if (sy0 + iy > p.sh) {
            // partial bottom row: reference count path
            area_fast_region(p, src, x0, x1, dy, dy + 1, dst + (dy - y0) * dstride, dstride);
            continue;
        }
        for (int r = 0; r < iy; r++) rows[r] = src.row(sy0 + r);
        const int lo = x0 * ix * cn, hi = xf1 * ix * cn;
        vsum_u16_avx2(rows.data(), iy, lo, hi, rs.data());
        uint8_t* D = dst + (dy - y0) * dstride;
        const int n = (xf1 - x0) * cn;
        const uint16_t* q = rs.data() + (size_t)x0 * ix * cn;
        int32_t* sm = sums.data();
        if (cn == 3 && ix == 2) {
            for (int dx = 0; dx < xf1 - x0; dx++, q += 6, sm += 3)
                sm[0] = q[0] + q[3], sm[1] = q[1] + q[4], sm[2] = q[2] + q[5];
        } else if (cn == 3 && ix == 4) {
            for (int dx = 0; dx < xf1 - x0; dx++, q += 12, sm += 3)
                sm[0] = q[0] + q[3] + q[6] + q[9], sm[1] = q[1] + q[4] + q[7] + q[10], sm[2] = q[2] + q[5] + q[8] + q[11];
        } else {
            for (int dx = 0; dx < xf1 - x0; dx++, q += ix * cn)
                for (int c = 0; c < cn; c++) {
                    int v = 0;
                    for (int k = 0; k < ix; k++) v += q[k * cn + c];
                    *sm++ = v;
                }
        }
        if (fast2)
            for (int e = 0; e < n; e++) D[e] = (uint8_t)((sums[e] + 2) >> 2);
        else
            cvt_scale_u8_avx2(sums.data(), p.fast_scale, D, n);
        if (xf1 < x1) area_fast_region(p, src, xf1, x1, dy, dy + 1, D + (size_t)(xf1 - x0) * cn, dstride);
    }
    return true;
#else
    return false;
#endif
}

#if defined(__x86_64__)
// linear horizontal pass: h[e] = S[i0]*w0 + S[i1]*w1 (int32) with window permutes (2 taps)
SHV_TARGET_AVX2 static void linear_h_avx2(const CvResizePlan& p, const uint8_t* S, int e0, int e1, int32_t* out) {
    const size_t np = p.ne_pad;
    const __m256i lo8 = _mm256_set1_epi32(0xff);
    for (int b = e0 / 8; b * 8 < e1; b++) {
        const int e = b * 8;
        __m256i acc = _mm256_setzero_si256();
        for (int j = 0; j < 2; j++) {
            __m256i v;
            if (p.b8_ok[b]) {
                __m256i win = _mm256_loadu_si256((const __m256i*)(S + p.b8_base[b]));
                __m256i dw = _mm256_loadu_si256((const __m256i*)(p.b8_dw.data() + j * np + e));
                __m256i sh = _mm256_loadu_si256((const __m256i*)(p.b8_sh.data() + j * np + e));
                v = _mm256_and_si256(_mm256_srlv_epi32(_mm256_permutevar8x32_epi32(win, dw), sh), lo8);
            } else {
                __m256i iv = _mm256_loadu_si256((const __m256i*)(p.eidx.data() + j * np + e));
                v = _mm256_and_si256(_mm256_i32gather_epi32((const int*)S, iv, 1), lo8);
            }
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(v, _mm256_loadu_si256((const __m256i*)(p.lw.data() + j * np + e))));
        }
        _mm256_storeu_si256((__m256i*)(out + (e - e0)), acc);
    }
}
SHV_TARGET_AVX2 static void linear_v_avx2(const int32_t* h0, const int32_t* h1, int b0, int b1, uint8_t* D, int n) {
    int i = 0;
    const __m256i B0 = _mm256_set1_epi32(b0), B1 = _mm256_set1_epi32(b1), two = _mm256_set1_epi32(2);
    for (; i + 16 <= n; i += 16) {
        __m256i r[2];
        for (int q = 0; q < 2; q++) {
            __m256i a = _mm256_loadu_si256((const __m256i*)(h0 + i + 8 * q));
            __m256i b = _mm256_loadu_si256((const __m256i*)(h1 + i + 8 * q));
            __m256i t = _mm256_add_epi32(_mm256_srai_epi32(_mm256_mullo_epi32(B0, _mm256_srai_epi32(a, 4)), 16),
                                         _mm256_srai_epi32(_mm256_mullo_epi32(B1, _mm256_srai_epi32(b, 4)), 16));
            r[q] = _mm256_srai_epi32(_mm256_add_epi32(t, two), 2);
        }
        __m256i w = _mm256_permute4x64_epi64(_mm256_packs_epi32(r[0], r[1]), 0xD8);
        _mm_storeu_si128((__m128i*)(D + i), _mm_packus_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256(w, 1)));
    }
    for (; i < n; i++) D[i] = (uint8_t)((((b0 * (h0[i] >> 4)) >> 16) + ((b1 * (h1[i] >> 4)) >> 16) + 2) >> 2);
}
#endif

static bool linear_region_vec(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1, uint8_t* dst,
                              ptrdiff_t dstride) {
#if defined(__x86_64__)
    if (isa() < SHV_ISA_AVX2 || p.lw.empty()) return false;
    const int cn = p.cn, e0 = x0 * cn, e1 = x1 * cn, n = e1 - e0;
    thread_local std::vector<int32_t> hb0, hb1;
    hb0.resize((size_t)n + 64);   // whole 8-element blocks are written: 16 elements of slack on each side
    hb1.resize((size_t)n + 64);
    int32_t* h0 = hb0.data() + 16;
    int32_t* h1 = hb1.data() + 16;
    int r0 = -1, r1 = -1;
    for (int dy = y0; dy < y1; dy++) {
        int sy0 = p.lyofs[dy];
        int ra = clipi(sy0, 0, p.sh), rb = clipi(sy0 + 1, 0, p.sh);
        if (ra != r0) {
            if (ra == r1) std::swap(h0, h1), std::swap(r0, r1);
            else linear_h_avx2(p, src.row(ra), e0, e1, h0), r0 = ra;
        }
        if (rb != r1) {
            if (rb == r0) memcpy(h1, h0, sizeof(int32_t) * (size_t)n);
            else linear_h_avx2(p, src.row(rb), e0, e1, h1);
            r1 = rb;
        }
        linear_v_avx2(h0, h1, p.lbeta[dy * 2], p.lbeta[dy * 2 + 1], dst + (dy - y0) * dstride, n);
    }
    return true;
#else
    return false;
#endif
}

static void area_fast_region(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1, uint8_t* dst,
                             ptrdiff_t dstride) {
    const int cn = p.cn, ix = p.iscale_x, iy = p.iscale_y;
    const int dwidth1 = p.sw / ix;  // pixels with a full block
    const bool fast2 = ix == 2 && iy == 2 && (cn == 1 || cn == 3);
    const float scale = p.fast_scale;
    std::vector<const uint8_t*> rows(iy);
    for (int dy = y0; dy < y1; dy++) {
        uint8_t* D = dst + (dy - y0) * dstride;
        int sy0 = dy * iy;
        if (sy0 >= p.sh) {
            memset(D, 0, (size_t)(x1 - x0) * cn);
            continue;
        }
        bool full_row = sy0 + iy <= p.sh;
        int nr = std::min(iy, p.sh - sy0);
        for (int r = 0; r < nr; r++) rows[r] = src.row(sy0 + r);
        for (int dx = x0; dx < x1; dx++) {
            for (int c = 0; c < cn; c++) {
                int sx0 = dx * ix;  // pixel
                uint8_t v;
                if (full_row && dx < dwidth1) {
                    int s = 0;
                    if (fast2) {
                        const uint8_t* S = rows[0] + (size_t)sx0 * cn + c;
                        const uint8_t* N = rows[1] + (size_t)sx0 * cn + c;
                        v = (uint8_t)((S[0] + S[cn] + N[0] + N[cn] + 2) >> 2);
                    } else {
                        for (int r = 0; r < iy; r++) {
                            const uint8_t* S = rows[r] + (size_t)sx0 * cn + c;
                            for (int q = 0; q < ix; q++) s += S[q * cn];
                        }
                        v = sat_u8_f((float)s * scale);
                    }
                } else {
                    int s = 0, count = 0;
                    if (sx0 >= p.sw) {
                        v = 0;
                    } else {
                        for (int r = 0; r < nr; r++) {
                            const uint8_t* S = rows[r] + (size_t)sx0 * cn + c;
                            for (int q = 0; q < ix; q++) {
                                if (sx0 + q >= p.sw) break;
                                s += S[q * cn];
                                count++;
                            }
                        }
                        v = sat_u8_f((float)s / count);
                    }
                }
                D[(dx - x0) * cn + c] = v;
            }
        }
    }
}

static void linear_region(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1, uint8_t* dst,
                          ptrdiff_t dstride) {
    const int cn = p.cn;
    const int n = (x1 - x0) * cn;
    std::vector<int> h0(n), h1(n);
    int r0_cached = -1, r1_cached = -1;
    auto hrow = [&](int sy, int* D) {
        const uint8_t* S = src.row(sy);
        for (int dx = x0; dx < x1; dx++) {
            int sx = p.lxofs[dx] * cn;
            int a0 = p.lalpha[dx * 2], a1 = p.lalpha[dx * 2 + 1];
            for (int c = 0; c < cn; c++) {
                int e = (dx - x0) * cn + c;
                if (dx < p.lxmax)
                    D[e] = S[sx + c] * a0 + S[sx + c + cn] * a1;
                else
                    D[e] = S[sx + c] * INTER_RESIZE_COEF_SCALE;
            }
        }
    };
    for (int dy = y0; dy < y1; dy++) {
        int sy0 = p.lyofs[dy];
        int ra = clipi(sy0, 0, p.sh), rb = clipi(sy0 + 1, 0, p.sh);
        // reuse computed rows like the reference ring (pure function of the row, so any reuse is exact)
        std::vector<int> t0, t1;
        if (ra == r0_cached) {
        } else if (ra == r1_cached) {
            std::swap(h0, h1);
            std::swap(r0_cached, r1_cached);
        } else {
            hrow(ra, h0.data());
            r0_cached = ra;
        }
        if (rb != r1_cached) {
            if (rb == r0_cached)
                h1 = h0;
            else
                hrow(rb, h1.data());
            r1_cached = rb;
        }
        int b0 = p.lbeta[dy * 2], b1 = p.lbeta[dy * 2 + 1];
        uint8_t* D = dst + (dy - y0) * dstride;
        for (int i = 0; i < n; i++)
            D[i] = (uint8_t)((((b0 * (h0[i] >> 4)) >> 16) + ((b1 * (h1[i] >> 4)) >> 16) + 2) >> 2);
    }
}

// INTER_LINEAR_EXACT, 1 channel, region entirely inside [minx,maxx) x [miny,maxy) (always true for downscaling):
// the reference keeps every intermediate bit (h = c0*S0 + c1*S1 < 2^16 exactly, res = h0*m0 + h1*m1 < 2^32 exactly,
// out = (res + 2^15) >> 16), so out = (sum_ij c_i m_j S_ij + 2^15) >> 16 and the passes may be swapped exactly:
// vertical first over contiguous source columns, then the 2-tap horizontal pass.
#if defined(__x86_64__)
SHV_TARGET_AVX2 static void linear_exact_c1_avx2(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1,
                                                 uint8_t* dst, ptrdiff_t dstride) {
    const int xlo = p.exofs[x0], xhi = std::min(p.exofs[x1 - 1] + 2, p.sw);
    thread_local std::vector<uint16_t> v;
    thread_local std::vector<int32_t> cpair;
    v.resize((size_t)p.sw + 32);
    cpair.resize((size_t)p.dw + 16);
    for (int dx = x0; dx < x1; dx++) cpair[dx] = (int32_t)(p.excoef[dx * 2] | ((uint32_t)p.excoef[dx * 2 + 1] << 16));
    const __m256i bias = _mm256_set1_epi32((1 << 23) + (1 << 15));
    const __m256i flip = _mm256_set1_epi32((int)0x80008000u);
    for (int dy = y0; dy < y1; dy++) {
        const uint8_t* S0 = src.row(p.eyofs[dy]);
        const uint8_t* S1 = src.row(p.eyofs[dy] + 1);
        const __m256i m0 = _mm256_set1_epi16((short)p.eycoef[dy * 2]), m1 = _mm256_set1_epi16((short)p.eycoef[dy * 2 + 1]);
        int x = xlo;
        for (; x + 16 <= xhi; x += 16) {
            __m256i a = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(S0 + x)));
            __m256i b = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(S1 + x)));
            __m256i r = _mm256_add_epi16(_mm256_mullo_epi16(a, m0), _mm256_mullo_epi16(b, m1));   // <= 65280, exact
            _mm256_storeu_si256((__m256i*)(v.data() + x), r);
        }
        for (; x < xhi; x++) v[x] = (uint16_t)(p.eycoef[dy * 2] * S0[x] + p.eycoef[dy * 2 + 1] * S1[x]);
        uint8_t* D = dst + (dy - y0) * dstride;
        int dx = x0;
        for (; dx + 8 <= x1; dx += 8) {
            __m256i ofs = _mm256_loadu_si256((const __m256i*)(p.exofs.data() + dx));
            // dword at v + ofs = v[o] | v[o+1] << 16; bias to signed (v - 32768) for pmaddwd, add 2^23 back
            __m256i g = _mm256_xor_si256(_mm256_i32gather_epi32((const int*)v.data(), ofs, 2), flip);
            __m256i r = _mm256_madd_epi16(g, _mm256_loadu_si256((const __m256i*)(cpair.data() + dx)));
            r = _mm256_srli_epi32(_mm256_add_epi32(r, bias), 16);
            __m128i w = _mm_packus_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
            _mm_storel_epi64((__m128i*)(D + (dx - x0)), _mm_packus_epi16(w, w));
        }
        for (; dx < x1; dx++) {
            int o = p.exofs[dx];
            D[dx - x0] = (uint8_t)(((uint32_t)p.excoef[dx * 2] * v[o] + (uint32_t)p.excoef[dx * 2 + 1] * v[o + 1] + 32768u) >> 16);
        }
    }
}
#endif

static void linear_exact_region(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1, uint8_t* dst,
                                ptrdiff_t dstride) {
    const int cn = p.cn;
    const int n = (x1 - x0) * cn;
    std::vector<uint16_t> h0(n), h1(n);
    int c0r = -1, c1r = -1;
    auto hrow = [&](int sy, uint16_t* D) {
        const uint8_t* S = src.row(sy);
        for (int dx = x0; dx < x1; dx++) {
            for (int c = 0; c < cn; c++) {
                int e = (dx - x0) * cn + c;
                if (dx < p.ex_minx)
                    D[e] = (uint16_t)(S[c] << 8);
                else if (dx >= p.ex_maxx)
                    D[e] = (uint16_t)(S[p.exofs[p.dw - 1] * cn + c] << 8);
                else {
                    int ix = p.exofs[dx] * cn + c;
                    D[e] = (uint16_t)(p.excoef[dx * 2] * S[ix] + p.excoef[dx * 2 + 1] * S[ix + cn]);
                }
            }
        }
    };
    for (int dy = y0; dy < y1; dy++) {
        uint8_t* D = dst + (dy - y0) * dstride;
        if (dy < p.ey_miny || dy >= p.ey_maxy) {
            int r = dy < p.ey_miny ? 0 : p.sh - 1;
            if (c0r != r) hrow(r, h0.data()), c0r = r;
            for (int i = 0; i < n; i++) D[i] = (uint8_t)((h0[i] + 128) >> 8);
            continue;
        }
        int ra = p.eyofs[dy], rb = ra + 1;
        if (ra == c1r) {
            std::swap(h0, h1);
            std::swap(c0r, c1r);
        } else if (ra != c0r) {
            hrow(ra, h0.data());
            c0r = ra;
        }
        if (rb != c1r) hrow(rb, h1.data()), c1r = rb;
        uint32_t m0 = p.eycoef[dy * 2], m1 = p.eycoef[dy * 2 + 1];
        for (int i = 0; i < n; i++) D[i] = (uint8_t)(((uint32_t)h0[i] * m0 + (uint32_t)h1[i] * m1 + 32768u) >> 16);
    }
}

void cv_resize_region(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1, uint8_t* dst,
                      ptrdiff_t dstride) {
    if (x1 <= x0 || y1 <= y0) return;
    switch (p.kind) {
        case 0:
            for (int y = y0; y < y1; y++) memcpy(dst + (y - y0) * dstride, src.row(y) + (size_t)x0 * p.cn,
                                                 (size_t)(x1 - x0) * p.cn);
            return;
        case 1: area_general_region(p, src, x0, x1, y0, y1, dst, dstride); return;
        case 2:
            if (!area_fast_region_vec(p, src, x0, x1, y0, y1, dst, dstride))
                area_fast_region(p, src, x0, x1, y0, y1, dst, dstride);
            return;
        case 3:
            if (!linear_region_vec(p, src, x0, x1, y0, y1, dst, dstride))
                linear_region(p, src, x0, x1, y0, y1, dst, dstride);
            return;
        case 4:
#if defined(__x86_64__)
            if (p.cn == 1 && isa() >= SHV_ISA_AVX2 && x0 >= p.ex_minx && x1 <= p.ex_maxx && y0 >= p.ey_miny &&
                y1 <= p.ey_maxy && p.sw >= 2) {
                linear_exact_c1_avx2(p, src, x0, x1, y0, y1, dst, dstride);
                return;
            }
#endif
            linear_exact_region(p, src, x0, x1, y0, y1, dst, dstride);
            return;
    }
}

// Contiguous image as a row source; rows are copied into a padded buffer when the gather kernels could read
// past the end of the image (last row).
struct ImgRows : SrcRows {
    const uint8_t* d;
    ptrdiff_t s;
    int h;
    size_t rb;
    std::vector<uint8_t> pad;
    ImgRows(const uint8_t* d_, ptrdiff_t s_, int h_, size_t rowbytes) : d(d_), s(s_), h(h_), rb(rowbytes) {}
    const uint8_t* row(int y) override {
        if (y == h - 1) {
            pad.assign(rb + 128, 0);
            memcpy(pad.data(), d + y * s, rb);
            return pad.data();
        }
        return d + y * s;
    }
};

static int resize_entry(const uint8_t* src, int sw, int sh, ptrdiff_t ss, int cn, uint8_t* dst, int dw, int dh,
                        ptrdiff_t ds, int interp) {
    if (!src || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || (cn != 1 && cn != 3)) {
        set_error("resize: bad arguments");
        return SHV_EINVAL;
    }
    CvResizePlan p;
    if (!cv_resize_plan(sw, sh, dw, dh, cn, interp, p)) {
        set_error("resize: unsupported combination (INTER_AREA enlarge?)");
        return SHV_EINVAL;
    }
    ImgRows rows(src, ss, sh, (size_t)sw * cn);
    cv_resize_region(p, rows, 0, dw, 0, dh, dst, ds);
    return 0;
}

}  // namespace shvp

extern "C" {
int shv_cv_resize_area(const uint8_t* src, int sw, int sh, ptrdiff_t ss, int cn, uint8_t* dst, int dw, int dh,
                       ptrdiff_t ds) {
    return shvp::resize_entry(src, sw, sh, ss, cn, dst, dw, dh, ds, 3);
}
int shv_cv_resize_linear(const uint8_t* src, int sw, int sh, ptrdiff_t ss, int cn, uint8_t* dst, int dw, int dh,
                         ptrdiff_t ds) {
    return shvp::resize_entry(src, sw, sh, ss, cn, dst, dw, dh, ds, 1);
}
int shv_cv_resize_linear_exact(const uint8_t* src, int sw, int sh, ptrdiff_t ss, int cn, uint8_t* dst, int dw,
                               int dh, ptrdiff_t ds) {
    return shvp::resize_entry(src, sw, sh, ss, cn, dst, dw, dh, ds, 5);
}
}
