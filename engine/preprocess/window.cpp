// Whole-window preprocessing (reference prepare_video given the 20 sampled native frames), per-frame face
// detection, and the per-participant cache.
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <memory>

#include "internal.hpp"


namespace shvp {

struct I420FullRows : RowSource {
    const shv_i420_image* im;
    std::vector<uint8_t> buf;
    static const int R = 20;   // the batched letterbox kernels hold 16 rows at once
    int ids[R];
    int next = 0;
    size_t rb;
    explicit I420FullRows(const shv_i420_image* i) : im(i) {
        rb = (size_t)im->width * 3 + 128;
        buf.assign(rb * R, 0);
        for (int k = 0; k < R; k++) ids[k] = -1;
    }
    const uint8_t* row(int y) override {
        for (int k = 0; k < R; k++)
            if (ids[k] == y) return buf.data() + rb * k;
        int k = next;
        next = (next + 1) % R;
        i420_row_to_rgb(im->y + y * im->y_stride, im->u + (y >> 1) * im->u_stride, im->v + (y >> 1) * im->v_stride, 0,
                        im->width, buf.data() + rb * k);
        ids[k] = y;
        return buf.data() + rb * k;
    }
};

struct RgbRowsW : RowSource {
    const uint8_t* d;
    ptrdiff_t s;
    const uint8_t* row(int y) override { return d + y * s; }
};

static int face_from_letterbox(const Cascade& c, const uint8_t* lb, shv_face* out) {
    uint8_t gray[320 * 180];
    rgb_to_gray(lb, 320 * 180, gray);
    std::vector<shv_rect> boxes;
    int rc = cascade_run(c, gray, 320, 180, 320, 1.05, 4, 15, 15, 0, 0, true, boxes);
    if (rc < 0) return rc;
    out->found = 0;
    out->box = shv_rect{0, 0, 0, 0};
    long best = -1;
    for (auto& b : boxes) {
        long a = (long)b.w * (long)b.h;
        if (a > best) best = a, out->box = b, out->found = 1;   // Python max(): first maximal element
    }
    return 0;
}

}  // namespace shvp

using namespace shvp;

static bool rgb_ok(const shv_rgb_image* f) { return f && f->data && f->width > 0 && f->height > 0 && f->stride >= 3 * f->width; }
static bool i420_ok(const shv_i420_image* f) {
    return f && f->y && f->u && f->v && f->width > 0 && f->height > 0 && f->y_stride >= f->width &&
           f->u_stride >= (f->width + 1) / 2 && f->v_stride >= (f->width + 1) / 2;
}

extern "C" int shv_letterbox_i420(const shv_i420_image* src, uint8_t* out) {
    if (!i420_ok(src) || !out) return (set_error("letterbox_i420: bad arguments"), SHV_EINVAL);
    I420FullRows rows(src);
    letterbox_run(letterbox_plan(src->width, src->height), rows, out);
    return 0;
}

extern "C" int shv_detect_face_letterboxed(const shv_cascade* c, const uint8_t* lb, shv_face* out) {
    if (!c || !lb || !out) return (set_error("detect_face: bad arguments"), SHV_EINVAL);
    return face_from_letterbox(c->c, lb, out);
}

extern "C" int shv_detect_face_rgb(const shv_cascade* c, const shv_rgb_image* f, shv_face* out) {
    if (!c || !rgb_ok(f) || !out) return (set_error("detect_face: bad arguments"), SHV_EINVAL);
    uint8_t lb[SHV_CROP_BYTES];
    RgbRowsW rows;
    rows.d = f->data;
    rows.s = f->stride;
    letterbox_run(letterbox_plan(f->width, f->height), rows, lb);
    return face_from_letterbox(c->c, lb, out);
}

extern "C" int shv_detect_face_i420(const shv_cascade* c, const shv_i420_image* f, shv_face* out) {
    if (!c || !i420_ok(f) || !out) return (set_error("detect_face: bad arguments"), SHV_EINVAL);
    uint8_t lb[SHV_CROP_BYTES];
    I420FullRows rows(f);
    letterbox_run(letterbox_plan(f->width, f->height), rows, lb);
    return face_from_letterbox(c->c, lb, out);
}

static const int kAnchor[5] = {0, 5, 10, 15, 19};

extern "C" int shv_window_prepare_rgb(const shv_cascade* c, const shv_rgb_image frames[20], uint8_t* crops,
                                      shv_window_info* info) {
    if (!c || !frames || !crops) return (set_error("window: bad arguments"), SHV_EINVAL);
    shv_window_info wi;
    for (int a = 0; a < 5; a++) {
        int rc = shv_detect_face_rgb(c, &frames[kAnchor[a]], &wi.faces[a]);
        if (rc < 0) return rc;
    }
    crop_geometry(wi.faces, wi.rects, wi.anchors);
    for (int i = 0; i < 20; i++) {
        int rc = shv_native_crop_rgb(&frames[i], wi.rects[i], crops + (size_t)i * SHV_CROP_BYTES);
        if (rc < 0) return rc;
    }
    if (info) *info = wi;
    return 0;
}

extern "C" int shv_window_prepare_i420(const shv_cascade* c, const shv_i420_image frames[20], uint8_t* crops,
                                       shv_window_info* info) {
    if (!c || !frames || !crops) return (set_error("window: bad arguments"), SHV_EINVAL);
    shv_window_info wi;
    for (int a = 0; a < 5; a++) {
        int rc = shv_detect_face_i420(c, &frames[kAnchor[a]], &wi.faces[a]);
        if (rc < 0) return rc;
    }
    crop_geometry(wi.faces, wi.rects, wi.anchors);
    for (int i = 0; i < 20; i++) {
        int rc = shv_native_crop_i420(&frames[i], wi.rects[i], crops + (size_t)i * SHV_CROP_BYTES);
        if (rc < 0) return rc;
    }
    if (info) *info = wi;
    return 0;
}

// ---------------------------------------------------------------- participant
namespace {
struct Slot {
    int64_t k = -1;
    double t = 0;
    int fmt = 0;   // 1 native rgb, 2 native i420, 3 'small' rgb (the reference's resized frame, <= 1280x720)
    int w = 0, h = 0;
    std::vector<uint8_t> data;
    shv_face face{};
};
}  // namespace

struct shv_participant {
    const shv_cascade* cascade = nullptr;
    shv_sampler sampler{};
    int hop = 4;
    std::map<int64_t, Slot> slots;          // sampled frames still cached, by target index
    std::map<int64_t, int64_t> missing;     // [first, first+count) runs of missing targets
    int64_t released_before = 0;            // frames with k < this were released
    int64_t windows_resolved = 0;           // windows [0, windows_resolved) are resolved
    int store = SHV_STORE_AUTO;
    size_t overhead = sizeof(shv_participant);
};

static bool is_missing(const shv_participant* p, int64_t k) {
    auto it = p->missing.upper_bound(k);
    if (it == p->missing.begin()) return false;
    --it;
    return k < it->first + it->second;
}

static int window_status(const shv_participant* p, int64_t w) {
    int64_t a = w * p->hop, b = a + 20;
    if (b > p->sampler.next_target) {
        // a missing target inside the already-resolved part makes it unavailable early
        for (int64_t k = a; k < p->sampler.next_target; k++)
            if (is_missing(p, k)) return SHV_WIN_UNAVAILABLE;
        return SHV_WIN_COLLECTING;
    }
    for (int64_t k = a; k < b; k++)
        if (is_missing(p, k)) return SHV_WIN_UNAVAILABLE;
    if (a < p->released_before) return SHV_WIN_EXPIRED;
    return SHV_WIN_READY;
}

static void add_missing(shv_participant* p, int64_t first, int64_t count) {
    if (count <= 0) return;
    p->missing[first] = std::max(p->missing.count(first) ? p->missing[first] : 0, count);
}

static void after_resolve(shv_participant* p, shv_push_result* res) {
    // windows whose last target is now resolved
    int64_t first_new = p->windows_resolved, last = -1;
    while ((p->windows_resolved * p->hop + 20) <= p->sampler.next_target) {
        last = p->windows_resolved;
        p->windows_resolved++;
    }
    if (res) {
        res->ready_window = last;
        res->ready_status = last >= 0 ? window_status(p, last) : -1;
    }
    (void)first_new;
    // keep the frames of the most recently resolved window (the caller extracts it after this push)
    if (last >= 0) shv_participant_release_before(p, last);
    // forget missing runs no future window can contain
    int64_t keep_from = (p->windows_resolved > 0 ? (p->windows_resolved - 1) * p->hop : 0);
    for (auto it = p->missing.begin(); it != p->missing.end();) {
        if (it->first + it->second <= keep_from)
            it = p->missing.erase(it);
        else
            ++it;
    }
}

extern "C" {

shv_participant* shv_participant_create(const shv_cascade* c, double start, int hop_targets) {
    if (!c) return (set_error("participant: no cascade"), nullptr);
    shv_participant* p = new shv_participant;
    p->cascade = c;
    p->hop = hop_targets > 0 ? hop_targets : 4;
    shv_sampler_init(&p->sampler, start);
    return p;
}

void shv_participant_free(shv_participant* p) { delete p; }

int shv_participant_set_store(shv_participant* p, int mode) {
    if (!p || mode < SHV_STORE_AUTO || mode > SHV_STORE_SMALL) return (set_error("participant: bad store mode"), SHV_EINVAL);
    p->store = mode;
    return 0;
}

static int push_common(shv_participant* p, double t, int fmt, const shv_rgb_image* rgb, const shv_i420_image* yuv,
                       shv_push_result* res) {
    int64_t mf = 0, mc = 0;
    int64_t k = shv_sampler_push(&p->sampler, t, &mf, &mc);
    if (res) {
        res->selected_target = k;
        res->missing_first = mf;
        res->missing_count = mc;
        res->ready_window = -1;
        res->ready_status = -1;
    }
    add_missing(p, mf, mc);
    if (k >= 0) {
        Slot s;
        s.k = k;
        s.t = t - p->sampler.origin;
        s.fmt = fmt;
        int rc;
        const int nw = fmt == 1 ? rgb->width : yuv->width, nh = fmt == 1 ? rgb->height : yuv->height;
        // face detection straight from the pushed frame (letterbox + gray + Haar, once per sampled frame)
        rc = fmt == 1 ? shv_detect_face_rgb(p->cascade, rgb, &s.face) : shv_detect_face_i420(p->cascade, yuv, &s.face);
        shv_canvas_info ci;
        const CvResizePlan* cp = canvas_plan(nw, nh, ci);
        const size_t native_bytes = fmt == 1 ? (size_t)nw * nh * 3 : (size_t)nw * nh + 2 * (size_t)((nw + 1) / 2) * ((nh + 1) / 2);
        const size_t small_bytes = (size_t)ci.new_w * ci.new_h * 3;
        bool small = cp && (p->store == SHV_STORE_SMALL || (p->store == SHV_STORE_AUTO && small_bytes < native_bytes));
        if (small) {
            // the reference's `small` = cv2.resize(native) (or native itself): cropping it with the copy plan and the
            // same paste offsets is exactly the canvas crop (canvas_info(small) == (copy, same left/top), checked)
            shv_canvas_info c2;
            small = canvas_info(ci.new_w, ci.new_h, c2) && c2.mode == 0 && c2.left == ci.left && c2.top == ci.top;
        }
        if (rc >= 0 && small) {
            s.fmt = 3;
            s.w = ci.new_w, s.h = ci.new_h;
            s.data.resize(small_bytes + 128);
            if (fmt == 1) {
                struct R : SrcRows {
                    const shv_rgb_image* f;
                    std::vector<uint8_t> pad;
                    const uint8_t* row(int y) override {
                        if (y == f->height - 1) {
                            pad.assign((size_t)f->width * 3 + 128, 0);
                            memcpy(pad.data(), f->data + y * f->stride, (size_t)f->width * 3);
                            return pad.data();
                        }
                        return f->data + y * f->stride;
                    }
                } rows;
                rows.f = rgb;
                cv_resize_region(*cp, rows, 0, ci.new_w, 0, ci.new_h, s.data.data(), (ptrdiff_t)ci.new_w * 3);
            } else {
                I420FullRows rows(yuv);
                struct R : SrcRows {
                    I420FullRows* r;
                    const uint8_t* row(int y) override { return r->row(y); }
                } rr;
                rr.r = &rows;
                cv_resize_region(*cp, rr, 0, ci.new_w, 0, ci.new_h, s.data.data(), (ptrdiff_t)ci.new_w * 3);
            }
        } else if (fmt == 1) {
            s.w = rgb->width, s.h = rgb->height;
            size_t rb = (size_t)s.w * 3;
            s.data.resize(rb * s.h + 128);
            for (int y = 0; y < s.h; y++) memcpy(s.data.data() + rb * y, rgb->data + y * rgb->stride, rb);
        } else {
            s.w = yuv->width, s.h = yuv->height;
            int cw = (s.w + 1) / 2, ch = (s.h + 1) / 2;
            s.data.resize((size_t)s.w * s.h + 2 * (size_t)cw * ch + 128);
            uint8_t* Y = s.data.data();
            uint8_t* U = Y + (size_t)s.w * s.h;
            uint8_t* V = U + (size_t)cw * ch;
            for (int y = 0; y < s.h; y++) memcpy(Y + (size_t)y * s.w, yuv->y + y * yuv->y_stride, s.w);
            for (int y = 0; y < ch; y++) {
                memcpy(U + (size_t)y * cw, yuv->u + y * yuv->u_stride, cw);
                memcpy(V + (size_t)y * cw, yuv->v + y * yuv->v_stride, cw);
            }
        }
        if (rc < 0) {
            add_missing(p, k, 1);   // detector failure -> windows with this target unavailable
        } else {
            p->slots[k] = std::move(s);
        }
    }
    after_resolve(p, res);
    return 0;
}

int shv_participant_push_rgb(shv_participant* p, double t, const shv_rgb_image* f, shv_push_result* res) {
    if (!p || !rgb_ok(f)) return (set_error("participant push: bad arguments"), SHV_EINVAL);
    return push_common(p, t, 1, f, nullptr, res);
}

int shv_participant_push_i420(shv_participant* p, double t, const shv_i420_image* f, shv_push_result* res) {
    if (!p || !i420_ok(f)) return (set_error("participant push: bad arguments"), SHV_EINVAL);
    return push_common(p, t, 2, nullptr, f, res);
}

void shv_participant_mark_missing(shv_participant* p, int64_t first, int64_t count) {
    if (p) add_missing(p, first, count);
}

int shv_participant_window_status(const shv_participant* p, int64_t w) {
    if (!p || w < 0) return SHV_EINVAL;
    return window_status(p, w);
}

int shv_participant_window_info(const shv_participant* p, int64_t w, shv_window_info* info, int64_t targets[20],
                                double times[20]) {
    if (!p || w < 0) return (set_error("participant: bad arguments"), SHV_EINVAL);
    if (window_status(p, w) != SHV_WIN_READY) return (set_error("participant: window %lld not ready", (long long)w), SHV_ESTATE);
    shv_window_info wi;
    int64_t a = w * p->hop;
    for (int i = 0; i < 5; i++) {
        auto it = p->slots.find(a + kAnchor[i]);
        if (it == p->slots.end()) return (set_error("participant: frame missing"), SHV_ESTATE);
        wi.faces[i] = it->second.face;
    }
    crop_geometry(wi.faces, wi.rects, wi.anchors);
    for (int i = 0; i < 20; i++) {
        auto it = p->slots.find(a + i);
        if (it == p->slots.end()) return (set_error("participant: frame missing"), SHV_ESTATE);
        if (targets) targets[i] = a + i;
        if (times) times[i] = it->second.t;
    }
    if (info) *info = wi;
    return 0;
}

int shv_participant_crop(const shv_participant* p, int64_t k, shv_rect rect, uint8_t* out) {
    if (!p || !out) return (set_error("participant crop: bad arguments"), SHV_EINVAL);
    auto it = p->slots.find(k);
    if (it == p->slots.end()) return (set_error("participant crop: frame %lld not cached", (long long)k), SHV_ESTATE);
    const Slot& s = it->second;
    if (s.fmt == 1 || s.fmt == 3) {
        shv_rgb_image im{s.data.data(), s.w, s.h, (ptrdiff_t)s.w * 3};
        return shv_native_crop_rgb(&im, rect, out);
    }
    int cw = (s.w + 1) / 2, ch = (s.h + 1) / 2;
    const uint8_t* Y = s.data.data();
    const uint8_t* U = Y + (size_t)s.w * s.h;
    const uint8_t* V = U + (size_t)cw * ch;
    shv_i420_image im{Y, U, V, s.w, cw, cw, s.w, s.h};
    return shv_native_crop_i420(&im, rect, out);
}

int shv_participant_window(const shv_participant* p, int64_t w, uint8_t* crops, shv_window_info* info,
                           int64_t targets[20], double times[20]) {
    shv_window_info wi;
    int rc = shv_participant_window_info(p, w, &wi, targets, times);
    if (rc < 0) return rc;
    for (int i = 0; i < 20; i++) {
        rc = shv_participant_crop(p, w * p->hop + i, wi.rects[i], crops + (size_t)i * SHV_CROP_BYTES);
        if (rc < 0) return rc;
    }
    if (info) *info = wi;
    return 0;
}

void shv_participant_release_before(shv_participant* p, int64_t w) {
    if (!p) return;
    int64_t k0 = w * p->hop;
    while (!p->slots.empty() && p->slots.begin()->first < k0) p->slots.erase(p->slots.begin());
    p->released_before = std::max(p->released_before, k0);
}

size_t shv_participant_bytes(const shv_participant* p) {
    if (!p) return 0;
    size_t b = p->overhead;
    for (auto& kv : p->slots) b += kv.second.data.capacity() + sizeof(Slot) + 48;   // + map node
    b += p->missing.size() * 64;
    return b;
}

int shv_participant_cached_frames(const shv_participant* p) { return p ? (int)p->slots.size() : 0; }
}
