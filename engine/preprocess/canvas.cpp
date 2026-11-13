// Native 1280x720 canvas (video_preprocess.py::_canvas) and the lazy face crop:
//   crop = cv2.resize(canvas[4y:4(y+h), 4x:4(x+w)], (320,180), INTER_AREA)
// Only the canvas pixels inside the patch are computed: each canvas pixel is a fixed function of a local
// native neighbourhood (see cvresize.cpp), so computing a sub-rectangle is exactly the full canvas restricted.
// Letterbox bars of the canvas are zeros, as np.zeros in the reference.
#include <algorithm>
#include <map>
#include <memory>

#include "internal.hpp"

namespace shvp {

const CvResizePlan* canvas_plan(int w, int h, shv_canvas_info& ci) {
    thread_local std::map<std::pair<int, int>, std::pair<shv_canvas_info, std::unique_ptr<CvResizePlan>>> cache;
    auto key = std::make_pair(w, h);
    auto it = cache.find(key);
    if (it != cache.end()) {
        ci = it->second.first;
        return it->second.second.get();
    }
    if (!canvas_info(w, h, ci)) return nullptr;
    std::unique_ptr<CvResizePlan> p(new CvResizePlan);
    bool ok = ci.mode == 0 ? cv_resize_plan(w, h, w, h, 3, 1, *p)
                           : cv_resize_plan(w, h, ci.new_w, ci.new_h, 3, ci.mode == 1 ? 3 : 1, *p);
    if (!ok) return nullptr;
    if (cache.size() > 32) cache.clear();
    CvResizePlan* raw = p.get();
    cache[key] = std::make_pair(ci, std::move(p));
    return raw;
}

const CvResizePlan& crop_plan(int rw, int rh) {
    thread_local std::map<std::pair<int, int>, std::unique_ptr<CvResizePlan>> cache;
    auto key = std::make_pair(rw, rh);
    auto it = cache.find(key);
    if (it != cache.end()) return *it->second;
    std::unique_ptr<CvResizePlan> p(new CvResizePlan);
    cv_resize_plan(4 * rw, 4 * rh, 320, 180, 3, 3, *p);
    CvResizePlan& ref = *p;
    cache[key] = std::move(p);
    return ref;
}

// Rows of a contiguous buffer; the last row is copied with slack so vector gathers may over-read 4 bytes.
struct PaddedRows : SrcRows {
    const uint8_t* d;
    ptrdiff_t s;
    int h;
    size_t rb;
    std::vector<uint8_t> pad;
    PaddedRows(const uint8_t* d_, ptrdiff_t s_, int h_, size_t rowbytes) : d(d_), s(s_), h(h_), rb(rowbytes) {}
    const uint8_t* row(int y) override {
        if (y == h - 1) {   // vector kernels read up to 64 bytes past the last needed byte
            if (pad.empty()) {
                pad.assign(rb + 128, 0);
                memcpy(pad.data(), d + y * s, rb);
            }
            return pad.data();
        }
        return d + y * s;
    }
};

void native_crop(int w, int h, SrcRows& native, shv_rect r, uint8_t* out) {
    shv_canvas_info ci;
    const CvResizePlan* cp = canvas_plan(w, h, ci);
    const int PW = 4 * r.w, PH = 4 * r.h;
    const int X0 = 4 * r.x, Y0 = 4 * r.y;
    thread_local std::vector<uint8_t> patch;
    patch.assign((size_t)PW * PH * 3 + 128, 0);
    // intersection of the patch with the pasted image
    int cx0 = std::max(X0, ci.left), cx1 = std::min(X0 + PW, ci.left + ci.new_w);
    int cy0 = std::max(Y0, ci.top), cy1 = std::min(Y0 + PH, ci.top + ci.new_h);
    if (cx1 > cx0 && cy1 > cy0) {
        uint8_t* dst = patch.data() + ((size_t)(cy0 - Y0) * PW + (cx0 - X0)) * 3;
        cv_resize_region(*cp, native, cx0 - ci.left, cx1 - ci.left, cy0 - ci.top, cy1 - ci.top, dst,
                         (ptrdiff_t)PW * 3);
    }
    // the patch buffer itself has 64 bytes of slack after the last row (gather over-read)
    struct PatchRows : SrcRows {
        const uint8_t* d;
        ptrdiff_t s;
        const uint8_t* row(int y) override { return d + y * s; }
    } pr;
    pr.d = patch.data();
    pr.s = (ptrdiff_t)PW * 3;
    const CvResizePlan& fp = crop_plan(r.w, r.h);
    cv_resize_region(fp, pr, 0, 320, 0, 180, out, 960);
}

// I420 rows converted on demand (only columns [lo, hi) of each requested row), small ring of rows.
struct I420Rows : SrcRows {
    const shv_i420_image* im;
    int lo, hi;
    static const int R = 8;
    std::vector<uint8_t> buf;
    int rowid[R];
    int next = 0;
    size_t rb;
    I420Rows(const shv_i420_image* i, int lo_, int hi_) : im(i), lo(lo_), hi(hi_) {
        rb = (size_t)im->width * 3 + 128;
        buf.assign(rb * R, 0);
        for (int k = 0; k < R; k++) rowid[k] = -1;
    }
    const uint8_t* row(int y) override {
        for (int k = 0; k < R; k++)
            if (rowid[k] == y) return buf.data() + rb * k;
        int k = next;
        next = (next + 1) % R;
        uint8_t* b = buf.data() + rb * k;
        i420_row_to_rgb(im->y + y * im->y_stride, im->u + (y >> 1) * im->u_stride, im->v + (y >> 1) * im->v_stride,
                        lo, hi, b + (size_t)lo * 3);
        rowid[k] = y;
        return b;
    }
};

}  // namespace shvp

using namespace shvp;

static bool rect_ok(shv_rect r) { return r.w > 0 && r.h > 0 && r.x >= 0 && r.y >= 0 && r.x + r.w <= 320 && r.y + r.h <= 180; }

extern "C" int shv_native_crop_rgb(const shv_rgb_image* src, shv_rect rect, uint8_t* out) {
    if (!src || !src->data || src->width <= 0 || src->height <= 0 || !out || !rect_ok(rect))
        return (set_error("native_crop: bad arguments"), SHV_EINVAL);
    shv_canvas_info ci;
    if (!canvas_plan(src->width, src->height, ci)) return (set_error("native_crop: frame size"), SHV_EINVAL);
    PaddedRows rows(src->data, src->stride, src->height, (size_t)src->width * 3);
    native_crop(src->width, src->height, rows, rect, out);
    return 0;
}

extern "C" int shv_native_crop_i420(const shv_i420_image* src, shv_rect rect, uint8_t* out) {
    if (!src || !src->y || src->width <= 0 || src->height <= 0 || !out || !rect_ok(rect))
        return (set_error("native_crop: bad arguments"), SHV_EINVAL);
    shv_canvas_info ci;
    const CvResizePlan* cp = canvas_plan(src->width, src->height, ci);
    if (!cp) return (set_error("native_crop: frame size"), SHV_EINVAL);
    // native columns needed by the canvas columns of this crop
    int cx0 = std::max(4 * rect.x, ci.left) - ci.left, cx1 = std::min(4 * (rect.x + rect.w), ci.left + ci.new_w) - ci.left;
    int lo = 0, hi = src->width;
    if (cx1 > cx0) cv_resize_src_cols(*cp, cx0, cx1, lo, hi);
    if (lo & 1) lo--;
    I420Rows rows(src, std::max(0, lo), std::min(src->width, hi));
    native_crop(src->width, src->height, rows, rect, out);
    return 0;
}

extern "C" int shv_canvas_rgb(const shv_rgb_image* src, uint8_t* canvas) {
    if (!src || !src->data || src->width <= 0 || src->height <= 0 || !canvas)
        return (set_error("canvas: bad arguments"), SHV_EINVAL);
    shv_canvas_info ci;
    const CvResizePlan* cp = canvas_plan(src->width, src->height, ci);
    if (!cp) return (set_error("canvas: frame size"), SHV_EINVAL);
    memset(canvas, 0, 1280 * 720 * 3);
    PaddedRows rows(src->data, src->stride, src->height, (size_t)src->width * 3);
    cv_resize_region(*cp, rows, 0, ci.new_w, 0, ci.new_h, canvas + ((size_t)ci.top * 1280 + ci.left) * 3, 3840);
    return 0;
}
