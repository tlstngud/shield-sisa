// Colour conversions.
//  * cv::cvtColor(RGB2GRAY), 8U, OpenCV 4.13: 15-bit fixed point (R*9798 + G*19235 + B*3735 + 16384) >> 15
//    (measured: the 14-bit 4899/9617/1868 variant of older releases differs on 43864 of 2^24 triples).
//  * ffmpeg 6.1 swscale unscaled yuv420p -> rgb24 (x86 SIMD path, libswscale/x86/yuv_2_rgb.asm; the same
//    arithmetic as the historic MMX yuv2rgb_template.c): with BT.601 limited-range coefficients from
//    ff_yuv2rgb_c_init_tables (contrast = saturation = 1<<16, brightness 0):
//      Y' = 8Y - yOffset (psubw), U' = sat16(8U - 1024), V' = sat16(8V - 1024) (psubsw)
//      y = pmulhw(Y', yCoeff); ub = pmulhw(U', ubCoeff); ug = pmulhw(U', ugCoeff); vg = pmulhw(V', vgCoeff);
//      vr = pmulhw(V', vrCoeff); cg = sat16(ug + vg)
//      R = packus(sat16(y + vr)), G = packus(sat16(y + cg)), B = packus(sat16(y + ub))
//    pmulhw(a, b) = (a*b) >> 16 (arithmetic).  Chroma of pixel (x, y) is U/V[y/2][x/2] (no interpolation).
#include <algorithm>

#include "internal.hpp"
#include "simd.hpp"

namespace shvp {

// ---------------------------------------------------------------- gray
static void gray_scalar(const uint8_t* rgb, size_t n, uint8_t* g) {
    for (size_t i = 0; i < n; i++) {
        int r = rgb[i * 3], gg = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
        g[i] = (uint8_t)((r * 9798 + gg * 19235 + b * 3735 + (1 << 14)) >> 15);
    }
}

#if defined(__x86_64__)
SHV_TARGET_AVX2 static void gray_avx2(const uint8_t* rgb, size_t n, uint8_t* g) {
    size_t i = 0;
    // 8 pixels per step: gather RGB triplets with a byte shuffle into 16-bit lanes, madd with coefficients.
    const __m128i shufRG = _mm_setr_epi8(0, -1, 1, -1, 3, -1, 4, -1, 6, -1, 7, -1, 9, -1, 10, -1);
    const __m128i shufB = _mm_setr_epi8(2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1);
    const __m128i cRG = _mm_setr_epi16(9798, 19235, 9798, 19235, 9798, 19235, 9798, 19235);
    const __m128i cB = _mm_set1_epi32(3735);
    const __m128i rnd = _mm_set1_epi32(1 << 14);
    for (; i + 6 <= n; i += 4) {   // reads 16 bytes (needs 12 + 4 slack -> require 6 pixels left)
        __m128i v = _mm_loadu_si128((const __m128i*)(rgb + i * 3));
        __m128i rg = _mm_madd_epi16(_mm_shuffle_epi8(v, shufRG), cRG);
        __m128i b = _mm_mullo_epi32(_mm_shuffle_epi8(v, shufB), cB);
        __m128i s = _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(rg, b), rnd), 15);
        s = _mm_packus_epi32(s, s);
        s = _mm_packus_epi16(s, s);
        uint32_t w = (uint32_t)_mm_cvtsi128_si32(s);
        memcpy(g + i, &w, 4);
    }
    gray_scalar(rgb + i * 3, n - i, g + i);
}
#endif

void rgb_to_gray(const uint8_t* rgb, size_t n, uint8_t* gray) {
#if defined(__x86_64__)
    if (isa() >= SHV_ISA_AVX2) return gray_avx2(rgb, n, gray);
#endif
    gray_scalar(rgb, n, gray);
}

// ---------------------------------------------------------------- I420 -> RGB24
struct Yuv2RgbCoeffs {
    int16_t y_coeff, vr, ub, ug, vg, y_off, u_off, v_off;
};
static int16_t round_to_int16(int64_t f) {
    int r = (int)((f + (1 << 15)) >> 16);
    if ((unsigned)(r + 0x8000) & ~0xFFFF) r = (r >> 31) ^ 0x7FFF;   // av_clip_int16
    return (int16_t)r;
}
static Yuv2RgbCoeffs make_coeffs() {
    // ff_yuv2rgb_coeffs[SWS_CS_ITU601] = {104597, 132201, 25675, 53279}, fullRange = 0
    const int inv_table[4] = {104597, 132201, 25675, 53279};
    int64_t crv = inv_table[0], cbu = inv_table[1], cgu = -inv_table[2], cgv = -inv_table[3];
    int64_t cy = 1 << 16, oy = 0;
    const int64_t contrast = 1 << 16, saturation = 1 << 16, brightness = 0;
    cy = (cy * 255) / 219;
    oy = 16 << 16;
    cy = (cy * contrast) >> 16;
    crv = (crv * contrast * saturation) >> 32;
    cbu = (cbu * contrast * saturation) >> 32;
    cgu = (cgu * contrast * saturation) >> 32;
    cgv = (cgv * contrast * saturation) >> 32;
    oy -= 256LL * brightness;
    Yuv2RgbCoeffs c;
    c.y_coeff = round_to_int16(cy * (1 << 13));
    c.vr = round_to_int16(crv * (1 << 13));
    c.ub = round_to_int16(cbu * (1 << 13));
    c.vg = round_to_int16(cgv * (1 << 13));
    c.ug = round_to_int16(cgu * (1 << 13));
    c.y_off = round_to_int16(oy * (1 << 3));
    c.u_off = c.v_off = 0x0400;
    return c;
}
static const Yuv2RgbCoeffs K = make_coeffs();

static inline int16_t sat16(int v) { return (int16_t)(v < -32768 ? -32768 : v > 32767 ? 32767 : v); }
static inline int mulhi(int16_t a, int16_t b) { return ((int)a * (int)b) >> 16; }

static void i420_row_scalar(const uint8_t* Y, const uint8_t* U, const uint8_t* V, int x0, int x1, uint8_t* rgb) {
    for (int x = x0; x < x1; x++) {
        int16_t yp = (int16_t)((Y[x] << 3) - K.y_off);
        int16_t up = sat16((U[x >> 1] << 3) - K.u_off);
        int16_t vp = sat16((V[x >> 1] << 3) - K.v_off);
        int y = mulhi(yp, K.y_coeff);
        int ub = mulhi(up, K.ub), ug = mulhi(up, K.ug), vg = mulhi(vp, K.vg), vr = mulhi(vp, K.vr);
        int cg = sat16(ug + vg);
        uint8_t* o = rgb + (size_t)(x - x0) * 3;
        o[0] = sat_u8_i(sat16(y + vr));
        o[1] = sat_u8_i(sat16(y + cg));
        o[2] = sat_u8_i(sat16(y + ub));
    }
}

#if defined(__x86_64__)
// 16 pixels per step (8 chroma samples).  Output interleave via byte shuffles.
SHV_TARGET_AVX2 static void i420_row_avx2(const uint8_t* Y, const uint8_t* U, const uint8_t* V, int x0, int x1,
                                          uint8_t* rgb) {
    int x = x0;
    if (x & 1) {   // align to an even pixel so chroma pairs line up
        i420_row_scalar(Y, U, V, x, x + 1, rgb);
        x++;
    }
    const __m256i yoff = _mm256_set1_epi16(K.y_off), uvoff = _mm256_set1_epi16(K.u_off);
    const __m256i cy = _mm256_set1_epi16(K.y_coeff), cvr = _mm256_set1_epi16(K.vr), cub = _mm256_set1_epi16(K.ub),
                  cug = _mm256_set1_epi16(K.ug), cvg = _mm256_set1_epi16(K.vg);
    alignas(32) uint8_t R[16], G[16], B[16];
    for (; x + 16 <= x1; x += 16) {
        __m256i y = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Y + x)));
        __m128i u8 = _mm_loadl_epi64((const __m128i*)(U + (x >> 1)));
        __m128i v8 = _mm_loadl_epi64((const __m128i*)(V + (x >> 1)));
        // duplicate each chroma sample for 2 pixels
        __m128i uu = _mm_unpacklo_epi8(u8, u8), vv = _mm_unpacklo_epi8(v8, v8);
        __m256i u = _mm256_cvtepu8_epi16(uu), v = _mm256_cvtepu8_epi16(vv);
        __m256i yp = _mm256_sub_epi16(_mm256_slli_epi16(y, 3), yoff);
        __m256i up = _mm256_subs_epi16(_mm256_slli_epi16(u, 3), uvoff);
        __m256i vp = _mm256_subs_epi16(_mm256_slli_epi16(v, 3), uvoff);
        __m256i yy = _mm256_mulhi_epi16(yp, cy);
        __m256i ub = _mm256_mulhi_epi16(up, cub), ug = _mm256_mulhi_epi16(up, cug);
        __m256i vg = _mm256_mulhi_epi16(vp, cvg), vr = _mm256_mulhi_epi16(vp, cvr);
        __m256i cg = _mm256_adds_epi16(ug, vg);
        __m256i r = _mm256_adds_epi16(yy, vr), g = _mm256_adds_epi16(yy, cg), b = _mm256_adds_epi16(yy, ub);
        __m128i r8 = _mm_packus_epi16(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
        __m128i g8 = _mm_packus_epi16(_mm256_castsi256_si128(g), _mm256_extracti128_si256(g, 1));
        __m128i b8 = _mm_packus_epi16(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1));
        _mm_store_si128((__m128i*)R, r8);
        _mm_store_si128((__m128i*)G, g8);
        _mm_store_si128((__m128i*)B, b8);
        uint8_t* o = rgb + (size_t)(x - x0) * 3;
        // interleave (shuffle-based)
        const __m128i m0r = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
        const __m128i m0g = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
        const __m128i m0b = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
        const __m128i m1r = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
        const __m128i m1g = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
        const __m128i m1b = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
        const __m128i m2r = _mm_setr_epi8(-1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1);
        const __m128i m2g = _mm_setr_epi8(-1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1);
        const __m128i m2b = _mm_setr_epi8(10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15);
        __m128i o0 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r8, m0r), _mm_shuffle_epi8(g8, m0g)),
                                  _mm_shuffle_epi8(b8, m0b));
        __m128i o1 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r8, m1r), _mm_shuffle_epi8(g8, m1g)),
                                  _mm_shuffle_epi8(b8, m1b));
        __m128i o2 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r8, m2r), _mm_shuffle_epi8(g8, m2g)),
                                  _mm_shuffle_epi8(b8, m2b));
        _mm_storeu_si128((__m128i*)o, o0);
        _mm_storeu_si128((__m128i*)(o + 16), o1);
        _mm_storeu_si128((__m128i*)(o + 32), o2);
    }
    (void)R, (void)G, (void)B;
    if (x < x1) i420_row_scalar(Y, U, V, x, x1, rgb + (size_t)(x - x0) * 3);
}
#endif

// rgb receives pixels [x0, x1) of the row (rgb points at pixel x0)
void i420_row_to_rgb(const uint8_t* y, const uint8_t* u, const uint8_t* v, int x0, int x1, uint8_t* rgb) {
#if defined(__x86_64__)
    if (isa() >= SHV_ISA_AVX2) return i420_row_avx2(y, u, v, x0, x1, rgb);
#endif
    i420_row_scalar(y, u, v, x0, x1, rgb);
}

}  // namespace shvp

extern "C" {
void shv_rgb_to_gray(const uint8_t* rgb, size_t n, uint8_t* gray) { shvp::rgb_to_gray(rgb, n, gray); }

int shv_i420_to_rgb24(const shv_i420_image* s, uint8_t* rgb, ptrdiff_t rgb_stride) {
    using namespace shvp;
    if (!s || !s->y || !s->u || !s->v || s->width <= 0 || s->height <= 0 || !rgb) {
        set_error("shv_i420_to_rgb24: bad arguments");
        return SHV_EINVAL;
    }
    for (int y = 0; y < s->height; y++)
        i420_row_to_rgb(s->y + y * s->y_stride, s->u + (y >> 1) * s->u_stride, s->v + (y >> 1) * s->v_stride, 0,
                        s->width, rgb + y * rgb_stride);
    return 0;
}
}
