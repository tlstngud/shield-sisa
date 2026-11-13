// Internal declarations of libshield_v_prep.  Exactness notes live next to each implementation.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "shv_prep.h"

namespace shvp {

// ---------------------------------------------------------------- errors / dispatch
void set_error(const char* fmt, ...);
int isa();                       // active level (1 scalar, 2 avx2, 3 avx512)
bool cpu_has_avx2();
bool cpu_has_avx512();
bool cpu_has_vbmi();

// ---------------------------------------------------------------- rounding helpers
// cvRound / Python round / numpy rint: round half to even (default FP environment is FE_TONEAREST).
static inline int round_even(double v) { return (int)std::nearbyint(v); }
static inline int round_even_f(float v) { return (int)std::nearbyintf(v); }
static inline int cv_floor(double v) { return (int)std::floor(v); }
static inline int cv_floor_f(float v) { return (int)std::floor(v); }
static inline int cv_ceil(double v) { return (int)std::ceil(v); }
static inline uint8_t sat_u8_i(int v) { return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
// saturate_cast<uchar>(float): cvRound then saturate
static inline uint8_t sat_u8_f(float v) { return sat_u8_i(round_even_f(v)); }

// ---------------------------------------------------------------- Pillow resample (8bpc, RGB)
struct PilCoeffs {
    int out_size = 0, ksize = 0;
    std::vector<int> bounds;     // xmin, count per output
    std::vector<int32_t> kk;     // fixed point (22 bits)
};
void pil_precompute(int in_size, float in0, float in1, int out_size, PilCoeffs& c);
struct LetterboxPlan {
    int src_w = 0, src_h = 0, new_w = 0, new_h = 0, left = 0, top = 0;
    bool need_h = false, need_v = false, copy = false;
    PilCoeffs h, v;
    int ybox_first = 0, ybox_last = 0;
};
const LetterboxPlan& letterbox_plan(int w, int h);   // cached per thread
// Row source: returns pointer to RGB row y of the native image (may convert on the fly).
struct RowSource {
    virtual const uint8_t* row(int y) = 0;
    virtual ~RowSource() {}
};
void letterbox_run(const LetterboxPlan& p, RowSource& src, uint8_t* out /*180x320x3*/);

// Horizontal / vertical Pillow kernels (dispatched)
typedef void (*pil_hrow_fn)(const uint8_t* src, uint8_t* dst, int out_w, const int* bounds, const int32_t* kk,
                            int ksize);
typedef void (*pil_vrow_fn)(const uint8_t* const* rows, int nrows, const int32_t* k, uint8_t* dst, int n_bytes);

// ---------------------------------------------------------------- OpenCV resize
struct AreaTab { int di, si; float alpha; };
int cv_area_tab(int ssize, int dsize, int cn, double scale, std::vector<AreaTab>& tab);

// Plan for an OpenCV resize (one source geometry -> one destination geometry), any region of the output can be
// produced exactly (each output pixel is a fixed function of a local neighbourhood).
struct CvResizePlan {
    int kind = 0;        // 0 copy, 1 area general, 2 area fast (integer), 3 linear (11-bit), 4 linear exact (8.8)
    int sw = 0, sh = 0, dw = 0, dh = 0, cn = 0;
    // area general
    std::vector<AreaTab> xtab, ytab;
    std::vector<int> xtab_first;   // per dx: first entry index in xtab (size dw+1, entries di == dx*cn)
    std::vector<int> ytab_first;   // per dy: first entry index in ytab (size dh+1)
    int xk = 0, yk = 0;            // max taps per output
    std::vector<int> xsi;          // padded taps: per dx xk source columns
    std::vector<float> xal;        // per dx xk weights (0 padded)
    std::vector<int> ysi;          // per dy yk source rows
    std::vector<float> yal;
    std::vector<int32_t> eidx;     // [xk][dw*cn] source byte offset of tap j for output element e (padded)
    std::vector<float> ew;         // [xk][dw*cn] weight (0 for padding taps)
    size_t ne_pad = 0;             // dw*cn rounded up to 16 (row length of eidx/ew/blk tables)
    // permute kernels: blocks of 8 elements read one 32-byte window (AVX2 vpermd + vpsrlvd),
    // blocks of 16 elements one 64-byte window (AVX-512 VBMI vpermb)
    std::vector<int32_t> b8_base, b8_dw, b8_sh;    // base per block; [xk][ne_pad] dword index / bit shift
    std::vector<uint8_t> b8_ok;
    std::vector<int32_t> b16_base;
    std::vector<uint8_t> b16_idx, b16_ok;           // [xk][ne_pad] byte index in the window
    std::vector<int32_t> lw;                        // linear: [2][ne_pad] integer tap weights (11-bit)
    // area fast
    int iscale_x = 0, iscale_y = 0;
    float fast_scale = 0.f;
    // linear (fixed point 11 bit)
    std::vector<int> lxofs;        // per dx source x (pixels)
    std::vector<int16_t> lalpha;   // per dx 2 coeffs
    std::vector<int> lyofs;        // per dy source row sy (unclamped)
    std::vector<int16_t> lbeta;    // per dy 2 coeffs
    int lxmax = 0;
    // linear exact
    std::vector<int> exofs;        // per dx
    std::vector<uint16_t> excoef;  // per dx 2
    int ex_minx = 0, ex_maxx = 0;
    std::vector<int> eyofs;
    std::vector<uint16_t> eycoef;
    int ey_miny = 0, ey_maxy = 0;
};
// interpolation: 1 = INTER_LINEAR, 3 = INTER_AREA, 5 = INTER_LINEAR_EXACT (OpenCV codes)
bool cv_resize_plan(int sw, int sh, int dw, int dh, int cn, int interpolation, CvResizePlan& p);
// Source access for region resizes: row(y) returns pointer to the first byte of source row y (full width).
struct SrcRows {
    virtual const uint8_t* row(int y) = 0;
    virtual ~SrcRows() {}
};
// Produce output pixels [x0,x1) x [y0,y1) into dst (row stride dstride, pixel (x0,y0) at dst).
void cv_resize_region(const CvResizePlan& p, SrcRows& src, int x0, int x1, int y0, int y1, uint8_t* dst,
                      ptrdiff_t dstride);
// Source rows needed for output rows [y0,y1) / source columns for output columns [x0,x1): [lo, hi)
void cv_resize_src_rows(const CvResizePlan& p, int y0, int y1, int& lo, int& hi);
void cv_resize_src_cols(const CvResizePlan& p, int x0, int x1, int& lo, int& hi);

// ---------------------------------------------------------------- colour
void rgb_to_gray(const uint8_t* rgb, size_t n, uint8_t* gray);
void i420_row_to_rgb(const uint8_t* y, const uint8_t* u, const uint8_t* v, int x0, int x1, uint8_t* rgb);

// ---------------------------------------------------------------- Haar
struct Cascade {
    int win_w = 0, win_h = 0;
    struct Stage { int first, ntrees; float threshold; };
    struct Stump { int feature; float threshold, left, right; };
    struct Feature { int nrects; int x[3], y[3], w[3], h[3]; float weight[3]; };
    std::vector<Stage> stages;
    std::vector<Stump> stumps;
    std::vector<Feature> features;
};
bool cascade_parse(const char* xml, size_t len, Cascade& c);
int cascade_run(const Cascade& c, const uint8_t* gray, int w, int h, ptrdiff_t stride, double scale_factor,
                int min_neighbors, int min_w, int min_h, int max_w, int max_h, bool group,
                std::vector<shv_rect>& out);

// per-scale feature layout prepared for an integral image row stride
struct HaarScaleCtx {
    int sum_stride = 0;                      // ints per integral row
    const int32_t* sum = nullptr;            // integral image (szi.h x sum_stride)
    const uint32_t* sq = nullptr;            // squared integral
    int szi_w = 0, szi_h = 0;
};
struct HaarFeatOfs {                          // offsets for one feature at a given stride
    int nrects;
    int o[3][4];
    float w[3];
};
// SIMD row kernels: evaluate stage 0 + variance for windows x = 0, step, 2*step, ... < xend in row y.
// Writes per position: code[i] = -1 (low variance / invalid), 0 (rejected at stage 0), 1 (passes stage 0);
// vnf[i] = variance normalisation factor (float).
struct HaarPrepared {
    std::vector<HaarFeatOfs> feat;           // per stump (in stage order)
    std::vector<float> thr, left, right;     // per stump
    std::vector<int> stage_first, stage_n;
    std::vector<float> stage_thr;
    int nofs[4];
    int stride;
    double area;
};
typedef void (*haar_stage0_fn)(const HaarPrepared& hp, const int32_t* sum, const uint32_t* sq, int y, int npos,
                               int step, int8_t* code, float* vnf);
typedef int (*haar_rest_fn)(const HaarPrepared& hp, const int32_t* sum, int y, const int* xs, const float* vnf,
                            int n, int* out_x);
// Block kernels: evaluate stages 1.. for the positions flagged alive[i] (row y, positions i*step), lanes in
// blocks of 8/16 consecutive positions with contiguous loads; returns passing x values in increasing order.
typedef int (*haar_block_fn)(const HaarPrepared& hp, const int32_t* sum, int y, int npos, int step,
                             const uint8_t* alive, const float* vnf, int* out_x);
#if defined(__x86_64__)
int haar_block_avx2(const HaarPrepared& hp, const int32_t* sum, int y, int npos, int step, const uint8_t* alive,
                    const float* vnf, int* out_x);
int haar_block_avx512(const HaarPrepared& hp, const int32_t* sum, int y, int npos, int step, const uint8_t* alive,
                      const float* vnf, int* out_x);
#endif
void haar_stage0_scalar(const HaarPrepared& hp, const int32_t* sum, const uint32_t* sq, int y, int npos, int step,
                        int8_t* code, float* vnf);
int haar_rest_scalar(const HaarPrepared& hp, const int32_t* sum, int y, const int* xs, const float* vnf, int n,
                     int* out_x);
#if defined(__x86_64__)
void haar_stage0_avx2(const HaarPrepared& hp, const int32_t* sum, const uint32_t* sq, int y, int npos, int step,
                      int8_t* code, float* vnf);
void haar_stage0_avx512(const HaarPrepared& hp, const int32_t* sum, const uint32_t* sq, int y, int npos, int step,
                        int8_t* code, float* vnf);
#endif
void integral_scalar(const uint8_t* img, int w, int h, ptrdiff_t stride, int32_t* sum, uint32_t* sq, int sstride);
#if defined(__x86_64__)
void integral_avx2(const uint8_t* img, int w, int h, ptrdiff_t stride, int32_t* sum, uint32_t* sq, int sstride);
void integral_avx512(const uint8_t* img, int w, int h, ptrdiff_t stride, int32_t* sum, uint32_t* sq, int sstride);
#endif

// ---------------------------------------------------------------- geometry / canvas
void crop_geometry(const shv_face faces[5], shv_rect rects[20], int& anchors);
bool canvas_info(int w, int h, shv_canvas_info& ci);
const CvResizePlan* canvas_plan(int w, int h, shv_canvas_info& ci);     // cached per thread
const CvResizePlan& crop_plan(int rw, int rh);                         // patch (4rw x 4rh) -> 320x180, cached
// generic lazy crop from a row source of the native frame
void native_crop(int w, int h, SrcRows& native, shv_rect rect, uint8_t* out);

}  // namespace shvp

struct shv_cascade {
    shvp::Cascade c;
};
