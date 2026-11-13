// SIMD Haar cascade kernels (AVX2: 8 windows per vector, AVX-512: 16).  Every lane performs exactly the scalar
// operation sequence of haar.cpp (OpenCV order):
//   rect sum (int32, ((p0-p1)-p2)+p3), cvt to float (exact, |sum| < 2^24), ret = w0*S0 + w1*S1 (+ w2*S2) in float,
//   * vnf (float); value < thr picks left/right (float), widened to double and added in stump order (double);
//   stage passes iff !(sum < stage_thr).  Variance: nf = area*sqsum - sum*sum (double, sqsum < 2^31 so the
//   signed conversion is exact), vnf = (float)(1/sqrt(nf)), valid iff nf > 0 && area*vnf < 0.1.
// Lanes past the end of a row read padding/next-row memory of the integral buffers; their results are dropped.
#include <algorithm>

#include "internal.hpp"
#include "simd.hpp"

#if defined(__x86_64__)
#pragma GCC push_options
#pragma GCC target("avx2,fma,bmi,bmi2,popcnt")
namespace shvp {

// ------------------------------------------------------------------ AVX2
// load 8 int32 corner values for windows x0, x0+step, ... (step 1 or 2)
static inline __m256i ld8(const int32_t* p, int step) {
    if (step == 1) return _mm256_loadu_si256((const __m256i*)p);
    __m256 a = _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i*)p));
    __m256 b = _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i*)(p + 8)));
    __m256 e = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));    // a0 a2 b0 b2 | a4 a6 b4 b6
    return _mm256_permute4x64_epi64(_mm256_castps_si256(e), 0xD8);  // a0 a2 a4 a6 b0 b2 b4 b6
}

static inline __m256i rsum8(const int32_t* p, const int* o, int step) {
    __m256i a = ld8(p + o[0], step), b = ld8(p + o[1], step), c = ld8(p + o[2], step), d = ld8(p + o[3], step);
    return _mm256_add_epi32(_mm256_sub_epi32(_mm256_sub_epi32(a, b), c), d);
}

static inline void stage_acc8(const HaarPrepared& hp, int s, const int32_t* pw, int step,
                                              __m256 vnf, __m256d& t0, __m256d& t1) {
    t0 = _mm256_setzero_pd();
    t1 = _mm256_setzero_pd();
    const int a = hp.stage_first[s], n = hp.stage_n[s];
    for (int i = a; i < a + n; i++) {
        const HaarFeatOfs& f = hp.feat[i];
        __m256 s0 = _mm256_cvtepi32_ps(rsum8(pw, f.o[0], step));
        __m256 s1 = _mm256_cvtepi32_ps(rsum8(pw, f.o[1], step));
        __m256 ret = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(f.w[0]), s0), _mm256_mul_ps(_mm256_set1_ps(f.w[1]), s1));
        if (f.w[2] != 0.0f) {
            __m256 s2 = _mm256_cvtepi32_ps(rsum8(pw, f.o[2], step));
            ret = _mm256_add_ps(ret, _mm256_mul_ps(_mm256_set1_ps(f.w[2]), s2));
        }
        __m256 val = _mm256_mul_ps(ret, vnf);
        __m256 lt = _mm256_cmp_ps(val, _mm256_set1_ps(hp.thr[i]), _CMP_LT_OQ);
        __m256 lr = _mm256_blendv_ps(_mm256_set1_ps(hp.right[i]), _mm256_set1_ps(hp.left[i]), lt);
        t0 = _mm256_add_pd(t0, _mm256_cvtps_pd(_mm256_castps256_ps128(lr)));
        t1 = _mm256_add_pd(t1, _mm256_cvtps_pd(_mm256_extractf128_ps(lr, 1)));
    }
}

static inline int stage_pass_mask8(__m256d t0, __m256d t1, double thr) {
    __m256d th = _mm256_set1_pd(thr);
    int m0 = _mm256_movemask_pd(_mm256_cmp_pd(t0, th, _CMP_NLT_UQ));   // !(t < thr)
    int m1 = _mm256_movemask_pd(_mm256_cmp_pd(t1, th, _CMP_NLT_UQ));
    return m0 | (m1 << 4);
}

void haar_stage0_avx2(const HaarPrepared& hp, const int32_t* sum, const uint32_t* sq, int y, int npos, int step,
                      int8_t* code, float* vnf_out) {
    struct K {
        static void run(const HaarPrepared& hp, const int32_t* sum, const uint32_t* sq, int y,
                                        int npos, int step, int8_t* code, float* vnf_out) {
            const int S = hp.stride;
            const __m256d area = _mm256_set1_pd(hp.area), tenth = _mm256_set1_pd(1e-1), zero = _mm256_setzero_pd();
            const __m256d one = _mm256_set1_pd(1.0);
            alignas(32) float vbuf[8];
            for (int i = 0; i < npos; i += 8) {
                const int x = i * step;
                const int32_t* pw = sum + (size_t)y * S + x;
                const int32_t* pq = (const int32_t*)sq + (size_t)y * S + x;
                __m256i vs = rsum8(pw, hp.nofs, step);
                __m256i vq = rsum8(pq, hp.nofs, step);   // wrapping uint32 arithmetic == int32 arithmetic
                __m256d s0 = _mm256_cvtepi32_pd(_mm256_castsi256_si128(vs));
                __m256d s1 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(vs, 1));
                __m256d q0 = _mm256_cvtepi32_pd(_mm256_castsi256_si128(vq));
                __m256d q1 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(vq, 1));
                __m256d nf0 = _mm256_sub_pd(_mm256_mul_pd(area, q0), _mm256_mul_pd(s0, s0));
                __m256d nf1 = _mm256_sub_pd(_mm256_mul_pd(area, q1), _mm256_mul_pd(s1, s1));
                __m256d pos0 = _mm256_cmp_pd(nf0, zero, _CMP_GT_OQ), pos1 = _mm256_cmp_pd(nf1, zero, _CMP_GT_OQ);
                __m256d r0 = _mm256_div_pd(one, _mm256_sqrt_pd(nf0)), r1 = _mm256_div_pd(one, _mm256_sqrt_pd(nf1));
                __m128 f0 = _mm256_cvtpd_ps(r0), f1 = _mm256_cvtpd_ps(r1);
                // valid: nf > 0 && area*(double)vnf < 0.1
                __m256d ok0 = _mm256_and_pd(pos0, _mm256_cmp_pd(_mm256_mul_pd(area, _mm256_cvtps_pd(f0)), tenth, _CMP_LT_OQ));
                __m256d ok1 = _mm256_and_pd(pos1, _mm256_cmp_pd(_mm256_mul_pd(area, _mm256_cvtps_pd(f1)), tenth, _CMP_LT_OQ));
                __m256 vnf = _mm256_set_m128(f1, f0);
                int pm = _mm256_movemask_pd(pos0) | (_mm256_movemask_pd(pos1) << 4);
                int okm = _mm256_movemask_pd(ok0) | (_mm256_movemask_pd(ok1) << 4);
                __m256d t0, t1;
                stage_acc8(hp, 0, pw, step, vnf, t0, t1);
                int pass = stage_pass_mask8(t0, t1, (double)hp.stage_thr[0]);
                _mm256_store_ps(vbuf, vnf);
                int lanes = std::min(8, npos - i);
                for (int l = 0; l < lanes; l++) {
                    bool okl = (okm >> l) & 1;
                    code[i + l] = okl ? (((pass >> l) & 1) ? 1 : 0) : -1;
                    vnf_out[i + l] = ((pm >> l) & 1) ? vbuf[l] : 1.f;
                }
            }
        }
    };
    K::run(hp, sum, sq, y, npos, step, code, vnf_out);
}

// block evaluation, contiguous loads
int haar_block_avx2(const HaarPrepared& hp, const int32_t* sum, int y, int npos, int step, const uint8_t* alive,
                    const float* vnf, int* out_x) {
    const int S = hp.stride;
    const int ns = (int)hp.stage_n.size();
    int m = 0;
    for (int i = 0; i < npos; i += 8) {
        uint64_t bits;
        memcpy(&bits, alive + i, 8);
        if (!bits) continue;
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(
            _mm256_cmpgt_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(alive + i))), _mm256_setzero_si256())));
        const int32_t* pw = sum + (size_t)y * S + (size_t)i * step;
        __m256 v = _mm256_loadu_ps(vnf + i);
        for (int s = 1; s < ns && mask; s++) {
            __m256d t0, t1;
            stage_acc8(hp, s, pw, step, v, t0, t1);
            mask &= stage_pass_mask8(t0, t1, (double)hp.stage_thr[s]);
        }
        while (mask) {
            int l = __builtin_ctz((unsigned)mask);
            mask &= mask - 1;
            out_x[m++] = (i + l) * step;
        }
    }
    return m;
}

}  // namespace shvp
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx2,fma,bmi,bmi2,popcnt")
namespace shvp {
// ------------------------------------------------------------------ AVX-512
static inline __m512i ld16(const int32_t* p, int step) {
    if (step == 1) return _mm512_loadu_si512((const void*)p);
    const __m512i even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    return _mm512_permutex2var_epi32(_mm512_loadu_si512((const void*)p), even, _mm512_loadu_si512((const void*)(p + 16)));
}
static inline __m512i rsum16(const int32_t* p, const int* o, int step) {
    __m512i a = ld16(p + o[0], step), b = ld16(p + o[1], step), c = ld16(p + o[2], step), d = ld16(p + o[3], step);
    return _mm512_add_epi32(_mm512_sub_epi32(_mm512_sub_epi32(a, b), c), d);
}
// stage accumulation for 16 lanes given a rect-sum functor
template <class RS>
static inline __mmask16 stage16(const HaarPrepared& hp, int s, __m512 vnf, RS rs) {
    __m512d t0 = _mm512_setzero_pd(), t1 = _mm512_setzero_pd();
    const int a = hp.stage_first[s], n = hp.stage_n[s];
    for (int i = a; i < a + n; i++) {
        const HaarFeatOfs& f = hp.feat[i];
        __m512 ret = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(f.w[0]), rs(f.o[0])),
                                   _mm512_mul_ps(_mm512_set1_ps(f.w[1]), rs(f.o[1])));
        if (f.w[2] != 0.0f) ret = _mm512_add_ps(ret, _mm512_mul_ps(_mm512_set1_ps(f.w[2]), rs(f.o[2])));
        __m512 val = _mm512_mul_ps(ret, vnf);
        __mmask16 lt = _mm512_cmp_ps_mask(val, _mm512_set1_ps(hp.thr[i]), _CMP_LT_OQ);
        __m512 lr = _mm512_mask_blend_ps(lt, _mm512_set1_ps(hp.right[i]), _mm512_set1_ps(hp.left[i]));
        t0 = _mm512_add_pd(t0, _mm512_cvtps_pd(_mm512_castps512_ps256(lr)));
        t1 = _mm512_add_pd(t1, _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(lr), 1))));
    }
    __m512d th = _mm512_set1_pd((double)hp.stage_thr[s]);
    __mmask8 p0 = _mm512_cmp_pd_mask(t0, th, _CMP_NLT_UQ), p1 = _mm512_cmp_pd_mask(t1, th, _CMP_NLT_UQ);
    return (__mmask16)(p0 | ((unsigned)p1 << 8));
}

void haar_stage0_avx512(const HaarPrepared& hp, const int32_t* sum, const uint32_t* sq, int y, int npos, int step,
                        int8_t* code, float* vnf_out) {
    struct K {
        static void run(const HaarPrepared& hp, const int32_t* sum, const uint32_t* sq, int y,
                                          int npos, int step, int8_t* code, float* vnf_out) {
            const int S = hp.stride;
            const __m512d area = _mm512_set1_pd(hp.area), tenth = _mm512_set1_pd(1e-1), zero = _mm512_setzero_pd();
            const __m512d one = _mm512_set1_pd(1.0);
            for (int i = 0; i < npos; i += 16) {
                const int x = i * step;
                const int32_t* pw = sum + (size_t)y * S + x;
                const int32_t* pq = (const int32_t*)sq + (size_t)y * S + x;
                __m512i vs = rsum16(pw, hp.nofs, step), vq = rsum16(pq, hp.nofs, step);
                __m512d s0 = _mm512_cvtepi32_pd(_mm512_castsi512_si256(vs));
                __m512d s1 = _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(vs, 1));
                __m512d q0 = _mm512_cvtepi32_pd(_mm512_castsi512_si256(vq));
                __m512d q1 = _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(vq, 1));
                __m512d nf0 = _mm512_sub_pd(_mm512_mul_pd(area, q0), _mm512_mul_pd(s0, s0));
                __m512d nf1 = _mm512_sub_pd(_mm512_mul_pd(area, q1), _mm512_mul_pd(s1, s1));
                __mmask8 pos0 = _mm512_cmp_pd_mask(nf0, zero, _CMP_GT_OQ), pos1 = _mm512_cmp_pd_mask(nf1, zero, _CMP_GT_OQ);
                __m256 f0 = _mm512_cvtpd_ps(_mm512_div_pd(one, _mm512_sqrt_pd(nf0)));
                __m256 f1 = _mm512_cvtpd_ps(_mm512_div_pd(one, _mm512_sqrt_pd(nf1)));
                __mmask8 ok0 = pos0 & _mm512_cmp_pd_mask(_mm512_mul_pd(area, _mm512_cvtps_pd(f0)), tenth, _CMP_LT_OQ);
                __mmask8 ok1 = pos1 & _mm512_cmp_pd_mask(_mm512_mul_pd(area, _mm512_cvtps_pd(f1)), tenth, _CMP_LT_OQ);
                __m512 vnf = _mm512_insertf32x8(_mm512_castps256_ps512(f0), f1, 1);
                __mmask16 pm = (__mmask16)(pos0 | ((unsigned)pos1 << 8));
                __mmask16 okm = (__mmask16)(ok0 | ((unsigned)ok1 << 8));
                vnf = _mm512_mask_blend_ps(pm, _mm512_set1_ps(1.f), vnf);
                __mmask16 pass = stage16(hp, 0, vnf, [&](const int* o) {
                    return _mm512_cvtepi32_ps(rsum16(pw, o, step));
                });
                int lanes = std::min(16, npos - i);
                __mmask16 lm = (__mmask16)((1u << lanes) - 1);
                // code = ok ? (pass ? 1 : 0) : -1
                __m512i c = _mm512_mask_blend_epi32(okm, _mm512_set1_epi32(-1),
                                                    _mm512_mask_blend_epi32(pass, _mm512_setzero_si512(), _mm512_set1_epi32(1)));
                _mm_mask_storeu_epi8(code + i, lm, _mm512_cvtepi32_epi8(c));
                _mm512_mask_storeu_ps(vnf_out + i, lm, vnf);
            }
        }
    };
    K::run(hp, sum, sq, y, npos, step, code, vnf_out);
}

int haar_block_avx512(const HaarPrepared& hp, const int32_t* sum, int y, int npos, int step, const uint8_t* alive,
                      const float* vnf, int* out_x) {
    const int S = hp.stride;
    const int ns = (int)hp.stage_n.size();
    int m = 0;
    for (int i = 0; i < npos; i += 16) {
        __m128i a = _mm_loadu_si128((const __m128i*)(alive + i));
        __mmask16 mask = _mm_cmpneq_epi8_mask(a, _mm_setzero_si128());
        if (!mask) continue;
        const int32_t* pw = sum + (size_t)y * S + (size_t)i * step;
        __m512 v = _mm512_loadu_ps(vnf + i);
        for (int s = 1; s < ns && mask; s++)
            mask &= stage16(hp, s, v, [&](const int* o) { return _mm512_cvtepi32_ps(rsum16(pw, o, step)); });
        unsigned mm = mask;
        while (mm) {
            int l = __builtin_ctz(mm);
            mm &= mm - 1;
            out_x[m++] = (i + l) * step;
        }
    }
    return m;
}

}  // namespace shvp
#pragma GCC pop_options

namespace shvp {
// ------------------------------------------------------------------ integral images (placeholder)
// sum/sq row y+1 = row y + running prefix of image row y.  The prefix is sequential per row; the row add is
// vectorised.  Wrapping uint32 arithmetic for the squared sums (as OpenCV's int32 sqsum).
void integral_avx2(const uint8_t* img, int w, int h, ptrdiff_t stride, int32_t* sum, uint32_t* sq, int S) {
    integral_scalar(img, w, h, stride, sum, sq, S);
}
void integral_avx512(const uint8_t* img, int w, int h, ptrdiff_t stride, int32_t* sum, uint32_t* sq, int S) {
    integral_scalar(img, w, h, stride, sum, sq, S);
}
}  // namespace shvp
#endif
