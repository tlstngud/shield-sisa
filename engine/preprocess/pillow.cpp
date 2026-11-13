// Pillow 11.3 Image.resize(BILINEAR) for 8-bit RGB (libImaging/Resample.c), and the 320x180 letterbox of
// shield_v/preprocess.py::letterbox_rgb.
//
// Exactness: coefficients are computed in double exactly as precompute_coeffs() (same expression order,
// -ffp-contract=off), converted with normalize_coeffs_8bpc() to 22-bit fixed point; the horizontal pass runs
// first on rows [ybox_first, ybox_last) and rounds to uint8 (clip8), then the vertical pass.  Integer
// accumulation (int32, no overflow for these magnitudes) is order-independent, so SIMD kernels are exact.
#include <algorithm>
#include <map>
#include <memory>

#include "internal.hpp"
#include "simd.hpp"

namespace shvp {

static const int PRECISION_BITS = 32 - 8 - 2;

static inline double bilinear_filter(double x) {
    if (x < 0.0) x = -x;
    if (x < 1.0) return 1.0 - x;
    return 0.0;
}

void pil_precompute(int inSize, float in0, float in1, int outSize, PilCoeffs& c) {
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, ksize, xmin, xmax;
    filterscale = scale = (double)(in1 - in0) / outSize;
    if (filterscale < 1.0) filterscale = 1.0;
    support = 1.0 * filterscale;          // BILINEAR support = 1.0
    ksize = (int)std::ceil(support) * 2 + 1;
    std::vector<double> kk((size_t)outSize * ksize, 0.0);
    c.bounds.assign((size_t)outSize * 2, 0);
    for (xx = 0; xx < outSize; xx++) {
        center = in0 + (xx + 0.5) * scale;
        ww = 0.0;
        ss = 1.0 / filterscale;
        xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        xmax = (int)(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        double* k = &kk[(size_t)xx * ksize];
        for (x = 0; x < xmax; x++) {
            double w = bilinear_filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0) k[x] /= ww;
        }
        for (; x < ksize; x++) k[x] = 0;
        c.bounds[xx * 2 + 0] = xmin;
        c.bounds[xx * 2 + 1] = xmax;
    }
    c.kk.resize(kk.size());
    for (size_t i = 0; i < kk.size(); i++) {
        if (kk[i] < 0)
            c.kk[i] = (int)(-0.5 + kk[i] * (1 << PRECISION_BITS));
        else
            c.kk[i] = (int)(0.5 + kk[i] * (1 << PRECISION_BITS));
    }
    c.ksize = ksize;
    c.out_size = outSize;
}

static inline uint8_t clip8(int in) {
    if (in >= (1 << PRECISION_BITS << 8)) return 255;
    if (in <= 0) return 0;
    return (uint8_t)(in >> PRECISION_BITS);
}

// ---------------------------------------------------------------- scalar kernels
static void pil_hrow_scalar(const uint8_t* src, uint8_t* dst, int out_w, const int* bounds, const int32_t* kk,
                            int ksize) {
    for (int xx = 0; xx < out_w; xx++) {
        int xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1];
        const int32_t* k = kk + (size_t)xx * ksize;
        int ss0 = 1 << (PRECISION_BITS - 1), ss1 = ss0, ss2 = ss0;
        const uint8_t* s = src + (size_t)xmin * 3;
        for (int x = 0; x < xmax; x++) {
            ss0 += s[x * 3 + 0] * k[x];
            ss1 += s[x * 3 + 1] * k[x];
            ss2 += s[x * 3 + 2] * k[x];
        }
        dst[xx * 3 + 0] = clip8(ss0);
        dst[xx * 3 + 1] = clip8(ss1);
        dst[xx * 3 + 2] = clip8(ss2);
    }
}

static void pil_vrow_scalar(const uint8_t* const* rows, int nrows, const int32_t* k, uint8_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        int ss = 1 << (PRECISION_BITS - 1);
        for (int y = 0; y < nrows; y++) ss += rows[y][i] * k[y];
        dst[i] = clip8(ss);
    }
}

// ---------------------------------------------------------------- SIMD kernels
#if defined(__x86_64__)
// Vertical: 8 (AVX2) / 16 (AVX-512) bytes per step, int32 accumulation.
SHV_TARGET_AVX2 static void pil_vrow_avx2(const uint8_t* const* rows, int nrows, const int32_t* k, uint8_t* dst,
                                          int n) {
    int i = 0;
    const __m256i half = _mm256_set1_epi32(1 << (PRECISION_BITS - 1));
    const __m256i maxv = _mm256_set1_epi32(255);
    for (; i + 16 <= n; i += 16) {
        __m256i a0 = half, a1 = half;
        for (int y = 0; y < nrows; y++) {
            __m128i b = _mm_loadu_si128((const __m128i*)(rows[y] + i));
            __m256i kv = _mm256_set1_epi32(k[y]);
            a0 = _mm256_add_epi32(a0, _mm256_mullo_epi32(_mm256_cvtepu8_epi32(b), kv));
            a1 = _mm256_add_epi32(a1, _mm256_mullo_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(b, 8)), kv));
        }
        // clip8: >=2^30 -> 255, <=0 -> 0, else >>22 (arithmetic shift of positive values)
        __m256i r0 = _mm256_srai_epi32(a0, PRECISION_BITS), r1 = _mm256_srai_epi32(a1, PRECISION_BITS);
        r0 = _mm256_min_epi32(_mm256_max_epi32(r0, _mm256_setzero_si256()), maxv);
        r1 = _mm256_min_epi32(_mm256_max_epi32(r1, _mm256_setzero_si256()), maxv);
        __m256i p = _mm256_packus_epi32(r0, r1);                 // 16-bit, lanes interleaved by 128
        p = _mm256_permute4x64_epi64(p, 0xD8);
        __m128i q = _mm_packus_epi16(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
        _mm_storeu_si128((__m128i*)(dst + i), q);
    }
    for (int j = i; j < n; j++) {
        int ss = 1 << (PRECISION_BITS - 1);
        for (int y = 0; y < nrows; y++) ss += rows[y][j] * k[y];
        dst[j] = clip8(ss);
    }
}

SHV_TARGET_AVX512 static void pil_vrow_avx512(const uint8_t* const* rows, int nrows, const int32_t* k, uint8_t* dst,
                                              int n) {
    int i = 0;
    const __m512i half = _mm512_set1_epi32(1 << (PRECISION_BITS - 1));
    for (; i + 32 <= n; i += 32) {
        __m512i a0 = half, a1 = half;
        for (int y = 0; y < nrows; y++) {
            __m256i b = _mm256_loadu_si256((const __m256i*)(rows[y] + i));
            __m512i kv = _mm512_set1_epi32(k[y]);
            a0 = _mm512_add_epi32(a0, _mm512_mullo_epi32(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(b)), kv));
            a1 = _mm512_add_epi32(a1, _mm512_mullo_epi32(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(b, 1)), kv));
        }
        __m512i r0 = _mm512_srai_epi32(a0, PRECISION_BITS), r1 = _mm512_srai_epi32(a1, PRECISION_BITS);
        // values >= 2^30 give >= 256 after the shift -> saturating narrow gives 255; negatives -> 0
        _mm_storeu_si128((__m128i*)(dst + i), _mm512_cvtusepi32_epi8(_mm512_max_epi32(r0, _mm512_setzero_si512())));
        _mm_storeu_si128((__m128i*)(dst + i + 16),
                         _mm512_cvtusepi32_epi8(_mm512_max_epi32(r1, _mm512_setzero_si512())));
    }
    for (int j = i; j < n; j++) {
        int ss = 1 << (PRECISION_BITS - 1);
        for (int y = 0; y < nrows; y++) ss += rows[y][j] * k[y];
        dst[j] = clip8(ss);
    }
}

// Horizontal: one output pixel per 128-bit lane group; taps gathered as 4 bytes (RGB + next byte, weight
// applied to all four lanes, lane 3 discarded).  Two output pixels per 256-bit vector.
SHV_TARGET_AVX2 static void pil_hrow_avx2(const uint8_t* src, uint8_t* dst, int out_w, const int* bounds,
                                          const int32_t* kk, int ksize) {
    const __m256i half = _mm256_set1_epi32(1 << (PRECISION_BITS - 1));
    int xx = 0;
    for (; xx + 2 <= out_w; xx += 2) {
        int xmin0 = bounds[xx * 2], n0 = bounds[xx * 2 + 1];
        int xmin1 = bounds[xx * 2 + 2], n1 = bounds[xx * 2 + 3];
        const int32_t* k0 = kk + (size_t)xx * ksize;
        const int32_t* k1 = k0 + ksize;
        const uint8_t* s0 = src + (size_t)xmin0 * 3;
        const uint8_t* s1 = src + (size_t)xmin1 * 3;
        __m256i acc = half;
        int n = n0 > n1 ? n0 : n1;
        for (int x = 0; x < n; x++) {
            int kv0 = x < n0 ? k0[x] : 0, kv1 = x < n1 ? k1[x] : 0;
            uint32_t p0 = 0, p1 = 0;
            if (x < n0) { p0 = s0[x * 3] | (s0[x * 3 + 1] << 8) | (s0[x * 3 + 2] << 16); }
            if (x < n1) { p1 = s1[x * 3] | (s1[x * 3 + 1] << 8) | (s1[x * 3 + 2] << 16); }
            __m256i pv = _mm256_cvtepu8_epi32(_mm_set_epi32(0, 0, (int)p1, (int)p0));
            // cvtepu8 expands 8 bytes: p0 bytes -> lanes 0..3, p1 bytes -> lanes 4..7
            __m256i kv = _mm256_setr_epi32(kv0, kv0, kv0, kv0, kv1, kv1, kv1, kv1);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(pv, kv));
        }
        alignas(32) int32_t r[8];
        _mm256_store_si256((__m256i*)r, acc);
        dst[xx * 3 + 0] = clip8(r[0]);
        dst[xx * 3 + 1] = clip8(r[1]);
        dst[xx * 3 + 2] = clip8(r[2]);
        dst[xx * 3 + 3] = clip8(r[4]);
        dst[xx * 3 + 4] = clip8(r[5]);
        dst[xx * 3 + 5] = clip8(r[6]);
    }
    if (xx < out_w) pil_hrow_scalar(src, dst + xx * 3, out_w - xx, bounds + xx * 2, kk + (size_t)xx * ksize, ksize);
}
#endif

#if defined(__x86_64__)
// ---------------------------------------------------------------- row-batched horizontal pass
// B source rows (8 AVX2, 16 AVX-512) are transposed to T[byte][B] so that tap j of output element e = (xx, c) is one
// contiguous B-byte load for all rows; per lane the exact scalar sum 2^21 + sum_j p_j*k_j (int32) is formed, then
// clip8.  Results of 4 consecutive elements are transposed back with packs + a byte shuffle.
SHV_TARGET_AVX2 static inline void tr8x16(const uint8_t* const* r, size_t off, uint8_t* T) {
    __m128i a0 = _mm_loadu_si128((const __m128i*)(r[0] + off)), a1 = _mm_loadu_si128((const __m128i*)(r[1] + off));
    __m128i a2 = _mm_loadu_si128((const __m128i*)(r[2] + off)), a3 = _mm_loadu_si128((const __m128i*)(r[3] + off));
    __m128i a4 = _mm_loadu_si128((const __m128i*)(r[4] + off)), a5 = _mm_loadu_si128((const __m128i*)(r[5] + off));
    __m128i a6 = _mm_loadu_si128((const __m128i*)(r[6] + off)), a7 = _mm_loadu_si128((const __m128i*)(r[7] + off));
    __m128i b0 = _mm_unpacklo_epi8(a0, a1), b1 = _mm_unpackhi_epi8(a0, a1);
    __m128i b2 = _mm_unpacklo_epi8(a2, a3), b3 = _mm_unpackhi_epi8(a2, a3);
    __m128i b4 = _mm_unpacklo_epi8(a4, a5), b5 = _mm_unpackhi_epi8(a4, a5);
    __m128i b6 = _mm_unpacklo_epi8(a6, a7), b7 = _mm_unpackhi_epi8(a6, a7);
    __m128i c0 = _mm_unpacklo_epi16(b0, b2), c1 = _mm_unpackhi_epi16(b0, b2);
    __m128i c2 = _mm_unpacklo_epi16(b1, b3), c3 = _mm_unpackhi_epi16(b1, b3);
    __m128i c4 = _mm_unpacklo_epi16(b4, b6), c5 = _mm_unpackhi_epi16(b4, b6);
    __m128i c6 = _mm_unpacklo_epi16(b5, b7), c7 = _mm_unpackhi_epi16(b5, b7);
    __m128i* o = (__m128i*)T;
    _mm_storeu_si128(o + 0, _mm_unpacklo_epi32(c0, c4));   // cols 0,1
    _mm_storeu_si128(o + 1, _mm_unpackhi_epi32(c0, c4));   // cols 2,3
    _mm_storeu_si128(o + 2, _mm_unpacklo_epi32(c1, c5));
    _mm_storeu_si128(o + 3, _mm_unpackhi_epi32(c1, c5));
    _mm_storeu_si128(o + 4, _mm_unpacklo_epi32(c2, c6));
    _mm_storeu_si128(o + 5, _mm_unpackhi_epi32(c2, c6));
    _mm_storeu_si128(o + 6, _mm_unpacklo_epi32(c3, c7));
    _mm_storeu_si128(o + 7, _mm_unpackhi_epi32(c3, c7));
}

SHV_TARGET_AVX2 static void pil_hbatch8_avx2(const uint8_t* const* rows, uint8_t* const* dst, int out_w,
                                             const int* bounds, const int32_t* kk, int ksize, int in_w) {
    const size_t nb = (size_t)in_w * 3;
    thread_local std::vector<uint8_t> Tv;
    Tv.resize(nb * 8 + 64);
    uint8_t* T = Tv.data();
    size_t off = 0;
    for (; off + 16 <= nb; off += 16) tr8x16(rows, off, T + off * 8);
    for (; off < nb; off++)
        for (int l = 0; l < 8; l++) T[off * 8 + l] = rows[l][off];
    const __m256i half = _mm256_set1_epi32(1 << (PRECISION_BITS - 1)), m255 = _mm256_set1_epi32(255);
    const __m256i zero = _mm256_setzero_si256();
    // per 128-bit lane after packs: bytes [e0 r0..3, e1 r0..3, e2 r0..3, e3 r0..3] -> [r0: e0..e3, r1: ...]
    const __m256i shuf = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                          0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const int ne = out_w * 3;
    int e = 0;
    __m256i acc[4];
    for (; e + 4 <= ne; e += 4) {
        for (int q = 0; q < 4; q++) {
            const int el = e + q, xx = el / 3, c = el - xx * 3;
            const int xmin = bounds[xx * 2], n = bounds[xx * 2 + 1];
            const int32_t* k = kk + (size_t)xx * ksize;
            const uint8_t* t = T + ((size_t)xmin * 3 + c) * 8;
            __m256i a = half;
            for (int j = 0; j < n; j++)
                a = _mm256_add_epi32(a, _mm256_mullo_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(t + j * 24))),
                                                           _mm256_set1_epi32(k[j])));
            a = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(a, PRECISION_BITS), zero), m255);
            acc[q] = a;
        }
        __m256i p01 = _mm256_packus_epi32(acc[0], acc[1]), p23 = _mm256_packus_epi32(acc[2], acc[3]);
        __m256i b = _mm256_shuffle_epi8(_mm256_packus_epi16(p01, p23), shuf);
        // lane 0: rows 0..3 (dword each), lane 1: rows 4..7
        alignas(32) uint32_t w[8];
        _mm256_store_si256((__m256i*)w, b);
        for (int l = 0; l < 8; l++) memcpy(dst[l] + e, &w[l], 4);
    }
    for (; e < ne; e++) {
        const int xx = e / 3, c = e - xx * 3;
        const int xmin = bounds[xx * 2], n = bounds[xx * 2 + 1];
        const int32_t* k = kk + (size_t)xx * ksize;
        for (int l = 0; l < 8; l++) {
            int ss = 1 << (PRECISION_BITS - 1);
            for (int j = 0; j < n; j++) ss += rows[l][(xmin + j) * 3 + c] * k[j];
            dst[l][e] = clip8(ss);
        }
    }
}

SHV_TARGET_AVX512 static void pil_hbatch16_avx512(const uint8_t* const* rows, uint8_t* const* dst, int out_w,
                                                  const int* bounds, const int32_t* kk, int ksize, int in_w) {
    const size_t nb = (size_t)in_w * 3;
    thread_local std::vector<uint8_t> Tv, Ta, Tb;
    Tv.resize(nb * 16 + 64);
    Ta.resize(nb * 8 + 64);
    Tb.resize(nb * 8 + 64);
    uint8_t* T = Tv.data();
    size_t off = 0;
    for (; off + 16 <= nb; off += 16) {
        tr8x16(rows, off, Ta.data());
        tr8x16(rows + 8, off, Tb.data());
        for (int i = 0; i < 8; i++) {
            __m128i a = _mm_loadu_si128((const __m128i*)(Ta.data() + i * 16));
            __m128i b = _mm_loadu_si128((const __m128i*)(Tb.data() + i * 16));
            _mm_storeu_si128((__m128i*)(T + (off + 2 * i) * 16), _mm_unpacklo_epi64(a, b));
            _mm_storeu_si128((__m128i*)(T + (off + 2 * i + 1) * 16), _mm_unpackhi_epi64(a, b));
        }
    }
    for (; off < nb; off++)
        for (int l = 0; l < 16; l++) T[off * 16 + l] = rows[l][off];
    const __m512i half = _mm512_set1_epi32(1 << (PRECISION_BITS - 1)), m255 = _mm512_set1_epi32(255);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i shuf = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
    const int ne = out_w * 3;
    int e = 0;
    __m512i acc[4];
    for (; e + 4 <= ne; e += 4) {
        for (int q = 0; q < 4; q++) {
            const int el = e + q, xx = el / 3, c = el - xx * 3;
            const int xmin = bounds[xx * 2], n = bounds[xx * 2 + 1];
            const int32_t* k = kk + (size_t)xx * ksize;
            const uint8_t* t = T + ((size_t)xmin * 3 + c) * 16;
            __m512i a = half;
            for (int j = 0; j < n; j++)
                a = _mm512_add_epi32(a, _mm512_mullo_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(t + j * 48))),
                                                           _mm512_set1_epi32(k[j])));
            acc[q] = _mm512_min_epi32(_mm512_max_epi32(_mm512_srai_epi32(a, PRECISION_BITS), zero), m255);
        }
        __m512i p01 = _mm512_packus_epi32(acc[0], acc[1]), p23 = _mm512_packus_epi32(acc[2], acc[3]);
        __m512i b = _mm512_shuffle_epi8(_mm512_packus_epi16(p01, p23), shuf);
        alignas(64) uint32_t w[16];
        _mm512_store_si512((void*)w, b);
        for (int l = 0; l < 16; l++) memcpy(dst[l] + e, &w[l], 4);
    }
    for (; e < ne; e++) {
        const int xx = e / 3, c = e - xx * 3;
        const int xmin = bounds[xx * 2], n = bounds[xx * 2 + 1];
        const int32_t* k = kk + (size_t)xx * ksize;
        for (int l = 0; l < 16; l++) {
            int ss = 1 << (PRECISION_BITS - 1);
            for (int j = 0; j < n; j++) ss += rows[l][(xmin + j) * 3 + c] * k[j];
            dst[l][e] = clip8(ss);
        }
    }
}
#endif

static pil_hrow_fn get_hrow() {
#if defined(__x86_64__)
    if (isa() >= SHV_ISA_AVX2) return pil_hrow_avx2;
#endif
    return pil_hrow_scalar;
}
static pil_vrow_fn get_vrow() {
#if defined(__x86_64__)
    if (isa() >= SHV_ISA_AVX512) return pil_vrow_avx512;
    if (isa() >= SHV_ISA_AVX2) return pil_vrow_avx2;
#endif
    return pil_vrow_scalar;
}

// ---------------------------------------------------------------- generic Pillow resize (ImagingResampleInner)
// Like Pillow: horizontal pass over source rows [ybox_first, ybox_last) into a temporary image, then the vertical
// pass.  Rows are batched for the SIMD horizontal kernels (each output byte is the same integer expression).
static void pil_resample(const PilCoeffs& ch, const PilCoeffs& cv, bool need_h, bool need_v, int in_w, int out_w,
                         int out_h, RowSource& src, uint8_t* dst, ptrdiff_t dstride) {
    pil_hrow_fn hrow = get_hrow();
    pil_vrow_fn vrow = get_vrow();
    const int ow = need_h ? out_w : in_w;
    const size_t rb = (size_t)ow * 3;
    int first, last;
    if (need_v) {
        first = cv.bounds[0];
        last = cv.bounds[out_h * 2 - 2] + cv.bounds[out_h * 2 - 1];
    } else {
        first = 0, last = out_h;
    }
    const int nrows = last - first;
    thread_local std::vector<uint8_t> tmpv;
    tmpv.resize(rb * nrows + 64);
    uint8_t* tmp = tmpv.data();
    int B = 1;
#if defined(__x86_64__)
    if (need_h) B = isa() >= SHV_ISA_AVX512 ? 16 : isa() >= SHV_ISA_AVX2 ? 8 : 1;
#endif
    const uint8_t* rp[16];
    uint8_t* dp[16];
    int r = first;
    if (B > 1) {
        for (; r + B <= last; r += B) {
            for (int i = 0; i < B; i++) {
                rp[i] = src.row(r + i);
                dp[i] = tmp + rb * (r + i - first);
            }
#if defined(__x86_64__)
            if (B == 16)
                pil_hbatch16_avx512(rp, dp, out_w, ch.bounds.data(), ch.kk.data(), ch.ksize, in_w);
            else
                pil_hbatch8_avx2(rp, dp, out_w, ch.bounds.data(), ch.kk.data(), ch.ksize, in_w);
#endif
        }
    }
    for (; r < last; r++) {
        const uint8_t* srow = src.row(r);
        if (need_h)
            hrow(srow, tmp + rb * (r - first), out_w, ch.bounds.data(), ch.kk.data(), ch.ksize);
        else
            memcpy(tmp + rb * (r - first), srow, rb);
    }
    if (!need_v) {
        for (int y = 0; y < out_h; y++) memcpy(dst + y * dstride, tmp + rb * y, rb);
        return;
    }
    std::vector<const uint8_t*> rows(cv.ksize);
    for (int yy = 0; yy < out_h; yy++) {
        int ymin = cv.bounds[yy * 2], cnt = cv.bounds[yy * 2 + 1];
        for (int y = 0; y < cnt; y++) rows[y] = tmp + rb * (ymin + y - first);
        vrow(rows.data(), cnt, cv.kk.data() + (size_t)yy * cv.ksize, dst + yy * dstride, ow * 3);
    }
}

// ---------------------------------------------------------------- letterbox plan
const LetterboxPlan& letterbox_plan(int w, int h) {
    thread_local std::map<std::pair<int, int>, std::unique_ptr<LetterboxPlan>> cache;
    auto key = std::make_pair(w, h);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    std::unique_ptr<LetterboxPlan> p(new LetterboxPlan);
    double scale = std::min(180.0 / h, 320.0 / w);
    p->src_w = w;
    p->src_h = h;
    p->new_w = std::max(1, round_even(w * scale));
    p->new_h = std::max(1, round_even(h * scale));
    p->left = (320 - p->new_w) / 2;
    p->top = (180 - p->new_h) / 2;
    p->copy = (p->new_w == w && p->new_h == h);
    p->need_h = p->new_w != w;
    p->need_v = p->new_h != h;
    if (!p->copy) {
        pil_precompute(w, 0.f, (float)w, p->new_w, p->h);
        pil_precompute(h, 0.f, (float)h, p->new_h, p->v);
        p->ybox_first = p->v.bounds[0];
        p->ybox_last = p->v.bounds[p->new_h * 2 - 2] + p->v.bounds[p->new_h * 2 - 1];
    }
    if (cache.size() > 64) cache.clear();
    auto& ref = *p;
    cache[key] = std::move(p);
    return ref;
}

void letterbox_run(const LetterboxPlan& p, RowSource& src, uint8_t* out) {
    memset(out, 0, 320 * 180 * 3);
    uint8_t* dst = out + ((size_t)p.top * 320 + p.left) * 3;
    if (p.copy) {
        for (int y = 0; y < p.src_h; y++) memcpy(dst + (size_t)y * 960, src.row(y), (size_t)p.src_w * 3);
        return;
    }
    pil_resample(p.h, p.v, p.need_h, p.need_v, p.src_w, p.new_w, p.new_h, src, dst, 960);
}

struct RgbRows : RowSource {
    const uint8_t* d;
    ptrdiff_t s;
    RgbRows(const uint8_t* d_, ptrdiff_t s_) : d(d_), s(s_) {}
    const uint8_t* row(int y) override { return d + y * s; }
};

}  // namespace shvp

extern "C" int shv_pil_resize_bilinear(const uint8_t* src, int sw, int sh, ptrdiff_t sstride, uint8_t* dst, int dw,
                                       int dh, ptrdiff_t dstride) {
    using namespace shvp;
    if (!src || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) {
        set_error("shv_pil_resize_bilinear: bad arguments");
        return SHV_EINVAL;
    }
    RgbRows rows(src, sstride);
    if (sw == dw && sh == dh) {
        for (int y = 0; y < sh; y++) memcpy(dst + y * dstride, src + y * sstride, (size_t)sw * 3);
        return 0;
    }
    PilCoeffs ch, cv;
    pil_precompute(sw, 0.f, (float)sw, dw, ch);
    pil_precompute(sh, 0.f, (float)sh, dh, cv);
    pil_resample(ch, cv, dw != sw, dh != sh, sw, dw, dh, rows, dst, dstride);
    return 0;
}

extern "C" int shv_letterbox_rgb(const shv_rgb_image* src, uint8_t* out) {
    using namespace shvp;
    if (!src || !src->data || src->width <= 0 || src->height <= 0 || !out) {
        set_error("shv_letterbox_rgb: bad arguments");
        return SHV_EINVAL;
    }
    RgbRows rows(src->data, src->stride);
    letterbox_run(letterbox_plan(src->width, src->height), rows, out);
    return 0;
}
