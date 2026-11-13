/*
 * shv_prep.h -- SHIELD-V native preprocessing (C API), libshield_v_prep.so
 *
 * Bit-exact re-implementation of the reference Python preprocessing of the SHIELD-V video detector
 * (reference/hf_model/video_preprocess.py + shield_v/preprocess.py) with the library versions it was
 * run with: Pillow 11.3 (Image.resize BILINEAR), OpenCV 4.13 (cvtColor RGB2GRAY, resize INTER_AREA /
 * INTER_LINEAR / INTER_LINEAR_EXACT, CascadeClassifier.detectMultiScale + groupRectangles), numpy 2.0
 * (np.interp, np.median, np.clip) and Python round() (half-to-even).  "Exact" below means: identical
 * uint8 output for identical input, on every dispatch level (scalar / AVX2 / AVX-512).
 *
 * Pipeline of one window (20 sampled frames), as the reference:
 *   native RGB24 frame --(Pillow BILINEAR letterbox)--> 320x180 RGB --(RGB2GRAY)--> Haar frontalface_default
 *   (scaleFactor 1.05, minNeighbors 4, minSize 15x15) on window indices 0,5,10,15,19 -> largest box (first of
 *   equal areas, OpenCV output order with cv::setNumThreads(1)) -> np.interp centres, crop size rule ->
 *   20 rectangles in 320x180 units -> native frame letterboxed to 1280x720 (INTER_AREA when shrinking,
 *   INTER_LINEAR when enlarging, plain copy when the size already fits) -> crop 4x rectangle ->
 *   INTER_AREA to 320x180.  Normalisation (/255, ImageNet mean/std) is NOT done here (the model libraries do it).
 *
 * Images: RGB24 interleaved, row stride in bytes (>= 3*width).  I420: planar Y (w x h), U, V ((w+1)/2 x (h+1)/2).
 * All functions return 0 on success and a negative SHV_E* code on failure (shv_prep_last_error() explains,
 * thread-local), unless documented otherwise.  All functions are thread-safe for distinct output buffers;
 * a shv_cascade is immutable after load and may be shared by any number of threads; a shv_participant is
 * single-threaded (one call at a time per participant).
 */
#ifndef SHV_PREP_H
#define SHV_PREP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#define SHV_PREP_API_VERSION 1
#define SHV_CROP_W 320
#define SHV_CROP_H 180
#define SHV_CANVAS_W 1280
#define SHV_CANVAS_H 720
#define SHV_WINDOW_FRAMES 20
#define SHV_NUM_ANCHORS 5
#define SHV_CROP_BYTES (SHV_CROP_W * SHV_CROP_H * 3)          /* 172800 */

/* error codes */
#define SHV_OK 0
#define SHV_EINVAL (-1)        /* bad argument */
#define SHV_ENOMEM (-2)
#define SHV_EIO (-3)           /* file / subprocess failure */
#define SHV_EFORMAT (-4)       /* cascade XML not understood */
#define SHV_EGAP (-5)          /* sampling: first frame at/after a target is more than 0.25 s late */
#define SHV_ESHORT (-6)        /* sampling: source too short / fewer than 20 frames */
#define SHV_EDUP (-7)          /* sampling: sampled PTS not distinct */
#define SHV_ESTATE (-8)        /* participant: window not ready / expired */

/* ------------------------------------------------------------------ dispatch / errors */
#define SHV_ISA_AUTO 0
#define SHV_ISA_SCALAR 1
#define SHV_ISA_AVX2 2         /* AVX2 + FMA3 + BMI2 (FMA is never used for arithmetic: -ffp-contract=off) */
#define SHV_ISA_AVX512 3       /* AVX-512 F/BW/DQ/VL */
/* Select the kernel level (SHV_ISA_AUTO = best supported; env SHV_PREP_ISA=scalar|avx2|avx512 overrides AUTO
 * at first use).  Returns the level now active, or SHV_EINVAL if the CPU does not support the request.
 * Outputs are identical on every level (tested).  Process-global; set it before starting worker threads. */
int shv_prep_set_isa(int isa);
int shv_prep_isa(void);                     /* active level */
int shv_prep_isa_supported(int isa);        /* 1/0 */
const char* shv_prep_isa_name(int isa);
const char* shv_prep_last_error(void);      /* thread-local message of the last failure */
const char* shv_prep_version(void);

/* ------------------------------------------------------------------ images */
typedef struct {
    const uint8_t* data;
    int width, height;
    ptrdiff_t stride;          /* bytes per row */
} shv_rgb_image;

typedef struct {
    const uint8_t *y, *u, *v;
    ptrdiff_t y_stride, u_stride, v_stride;
    int width, height;          /* luma size; chroma is ((w+1)/2, (h+1)/2) */
} shv_i420_image;

typedef struct { int x, y, w, h; } shv_rect;

/* ------------------------------------------------------------------ frame sampling
 * Offline, exactly shield_v/preprocess.py::sample_frames as run through reference/shim/av.py:
 *   pts[0..n) are the stream's presentation timestamps in presentation order (sorted, unique; packets with the
 *   discard flag removed), time base tb_num/tb_den.  origin = pts[0]*tb (the first decoded frame); time_i =
 *   float(pts_i*tb) - origin; for target_j = window_start + j/2 (j = 0..19) the sampled frame is the first frame
 *   with time + 1e-6 >= target_j (frames already used are not reused); if that frame's time - target_j > 0.25 the
 *   window fails with SHV_EGAP.  source_duration (seconds, as the reference computes it: stream duration_ts*tb,
 *   else container duration) must satisfy source_duration + 1e-3 >= window_start + 10, else SHV_ESHORT; pass
 *   a NaN to skip that check.  Fewer than 20 frames -> SHV_ESHORT.
 *   Outputs: out_index[20] (indices into pts), out_times[20] (seconds relative to origin).
 */
/* float(pts * Fraction(num, den)) of Python: the exact rational rounded to the nearest double (ties even). */
double shv_pts_seconds(int64_t pts, int64_t tb_num, int64_t tb_den);
int shv_sample_window(const int64_t* pts, size_t n, int64_t tb_num, int64_t tb_den, double window_start,
                      double source_duration, size_t out_index[20], double out_times[20]);

/* Live sampler (per stream).  Targets are t_k = start + k*0.5 s (k = 0,1,2,...) relative to the first pushed
 * frame.  Per target the selected frame is the first frame with rel_time + 1e-6 >= t_k; if that frame is more
 * than 0.25 s late the target is MISSING (every window that contains it is unavailable) and the same frame is
 * then tried against the following targets.  This is the per-target form of sample_frames: for strictly
 * increasing times the frame chosen for target k does not depend on the window start (proof in
 * docs/PERFORMANCE.md), so window w (targets 4w .. 4w+19 with the 2 s hop) equals sample_frames(start + 2w)
 * up to the floating-point evaluation of the target time (start + k*0.5 here). */
typedef struct {
    double start;              /* first target, seconds after the first frame (0 for live streams) */
    double origin;             /* time of the first pushed frame (set by the first push) */
    int64_t next_target;       /* next unresolved target index */
    int has_origin;
} shv_sampler;
void shv_sampler_init(shv_sampler* s, double start);
/* Feed the next frame time t (seconds, strictly increasing).  Returns the target index k this frame was
 * selected for, or -1 if the frame is not sampled.  *missing_first / *missing_count (may be NULL) receive the
 * range of targets declared MISSING by this call (count 0 if none). */
int64_t shv_sampler_push(shv_sampler* s, double t, int64_t* missing_first, int64_t* missing_count);

/* ------------------------------------------------------------------ components */
/* Pillow 11.3 Image.resize((round(w*s), round(h*s)), BILINEAR) with s = min(180/h, 320/w), pasted centred on a
 * black 320x180 canvas: shield_v/preprocess.py::letterbox_rgb.  out: 180 x 320 x 3 (stride 960). */
int shv_letterbox_rgb(const shv_rgb_image* src, uint8_t* out);
/* Same for an I420 frame; equals shv_i420_to_rgb24 followed by shv_letterbox_rgb (rows converted on the fly). */
int shv_letterbox_i420(const shv_i420_image* src, uint8_t* out);
/* cv2.cvtColor(RGB2GRAY) for 8U (OpenCV 4.13): (9798 R + 19235 G + 3735 B + 16384) >> 15. */
void shv_rgb_to_gray(const uint8_t* rgb, size_t npixels, uint8_t* gray);
/* I420 (BT.601 limited range) -> RGB24 exactly as ffmpeg 6.1 swscale does for
 * `scale=in_color_matrix=bt601:in_range=auto:flags=bilinear,format=rgb24` from yuv420p on x86-64
 * (the unscaled SSSE3 yuv2rgb path; chroma replicated per 2x2 block).  See docs/PERFORMANCE.md for the
 * measured parity. */
int shv_i420_to_rgb24(const shv_i420_image* src, uint8_t* rgb, ptrdiff_t rgb_stride);

/* Haar cascade (OpenCV 4.13 CascadeClassifier, new-format XML, stump trees, upright HAAR features). */
typedef struct shv_cascade shv_cascade;
/* path NULL -> the haarcascade_frontalface_default.xml embedded in the library (engine/preprocess/data/). */
shv_cascade* shv_cascade_load(const char* path);
void shv_cascade_free(shv_cascade* c);
/* detectMultiScale(gray, scaleFactor, minNeighbors, flags=0, minSize, maxSize) on an 8-bit image, including
 * the scale pyramid (INTER_LINEAR_EXACT), integral / squared integral images, variance normalisation and
 * low-variance rejection, stage/stump evaluation, window stepping with the stage-0 skip, stripe rounding,
 * groupRectangles(minNeighbors, 0.2) and clipping to the image.  Boxes are written in OpenCV's output order
 * for cv::setNumThreads(1) (the reference ran with 1 thread).  max_w/max_h 0 = image size.
 * Returns the number of boxes (all of them; at most max_out are written) or a negative error. */
int shv_cascade_detect(const shv_cascade* c, const uint8_t* gray, int width, int height, ptrdiff_t stride,
                       double scale_factor, int min_neighbors, int min_w, int min_h, int max_w, int max_h,
                       shv_rect* out, int max_out);
/* Raw candidates before grouping (minNeighbors = 0 in OpenCV), same order.  For tests. */
int shv_cascade_candidates(const shv_cascade* c, const uint8_t* gray, int width, int height, ptrdiff_t stride,
                           double scale_factor, int min_w, int min_h, int max_w, int max_h,
                           shv_rect* out, int max_out);

/* One anchor observation of the reference: frame index in the window (0,5,10,15,19), largest box. */
typedef struct {
    int found;                 /* 0: no face on this anchor frame */
    shv_rect box;              /* largest detection (first of equal area) in 320x180 letterbox coordinates */
} shv_face;
/* letterbox -> gray -> detect(1.05, 4, 15x15) -> largest box, for one frame (what the reference does per anchor). */
int shv_detect_face_rgb(const shv_cascade* c, const shv_rgb_image* frame, shv_face* out);
int shv_detect_face_i420(const shv_cascade* c, const shv_i420_image* frame, shv_face* out);
/* Same starting from an already letterboxed 180x320x3 frame. */
int shv_detect_face_letterboxed(const shv_cascade* c, const uint8_t* letterbox_rgb, shv_face* out);

/* video_preprocess.py::_boxes geometry: faces[5] are the anchors of window indices 0,5,10,15,19.
 * rects[20] receive (x, y, w, h) in 320x180 units; returns the anchor count (0..5).  Zero anchors -> the
 * reference whole-frame fallback (0,0,320,180). */
int shv_crop_geometry(const shv_face faces[5], shv_rect rects[20]);

/* Native canvas geometry of video_preprocess.py::_canvas for a w x h frame. */
typedef struct {
    int src_w, src_h;
    int new_w, new_h;          /* letterboxed size inside 1280x720 */
    int left, top;             /* paste offset */
    int mode;                  /* 0 copy, 1 INTER_AREA (shrink), 2 INTER_LINEAR (enlarge) */
} shv_canvas_info;
int shv_canvas_info_get(int width, int height, shv_canvas_info* info);
/* Full 1280x720x3 canvas (reference _canvas).  For tests; the crop functions below never build it. */
int shv_canvas_rgb(const shv_rgb_image* src, uint8_t* canvas /* 720*1280*3 */);
/* Lazy native crop: cv2.resize(canvas[4y:4(y+h), 4x:4(x+w)], (320,180), INTER_AREA) computing only the canvas
 * pixels inside the crop (and, for I420 input, converting only the native pixels those need).  Exactly equal
 * to building the full canvas.  rect must satisfy 0<=x, 0<=y, x+w<=320, y+h<=180, w,h>0.  out: 180x320x3. */
int shv_native_crop_rgb(const shv_rgb_image* src, shv_rect rect, uint8_t* out);
int shv_native_crop_i420(const shv_i420_image* src, shv_rect rect, uint8_t* out);

/* Low-level OpenCV / Pillow resize re-implementations (tests).  cn = 1 or 3. */
int shv_cv_resize_area(const uint8_t* src, int sw, int sh, ptrdiff_t sstride, int cn,
                       uint8_t* dst, int dw, int dh, ptrdiff_t dstride);
int shv_cv_resize_linear(const uint8_t* src, int sw, int sh, ptrdiff_t sstride, int cn,
                         uint8_t* dst, int dw, int dh, ptrdiff_t dstride);
int shv_cv_resize_linear_exact(const uint8_t* src, int sw, int sh, ptrdiff_t sstride, int cn,
                               uint8_t* dst, int dw, int dh, ptrdiff_t dstride);
int shv_pil_resize_bilinear(const uint8_t* src, int sw, int sh, ptrdiff_t sstride,      /* RGB only */
                            uint8_t* dst, int dw, int dh, ptrdiff_t dstride);

/* ------------------------------------------------------------------ whole window (reference prepare_video) */
typedef struct {
    int anchors;               /* face_detection_anchors, 0..5 */
    shv_face faces[5];         /* per anchor frame (window indices 0,5,10,15,19) */
    shv_rect rects[20];        /* crop rectangles in 320x180 units */
} shv_window_info;
/* frames[20]: the 20 sampled native frames of the window (any size, may differ).  Letterbox+Haar run only on
 * the 5 anchor frames (the reference letterboxes all 20 but uses only these).  crops: 20 x 180 x 320 x 3. */
int shv_window_prepare_rgb(const shv_cascade* c, const shv_rgb_image frames[20], uint8_t* crops,
                           shv_window_info* info);
int shv_window_prepare_i420(const shv_cascade* c, const shv_i420_image frames[20], uint8_t* crops,
                            shv_window_info* info);

/* ------------------------------------------------------------------ per-participant helper (live)
 * Push every decoded frame of one participant (presentation order, t = PTS in seconds, strictly increasing).  The
 * built-in shv_sampler picks the sampled frames (targets t_k = start + k/2 s after the first frame); for each sampled
 * frame the helper runs letterbox + gray + Haar ONCE (every sampled frame is an anchor of one or two windows of the
 * 2 s hop: k = 4w, 4w+5, 4w+10, 4w+15, 4w+19) and keeps a private copy of the frame (see storage modes).  Window w
 * covers targets hop*w .. hop*w+19 (hop_targets = 4 -> 2 s); with start = 0 its frames and crops are exactly
 * video_preprocess.prepare_video(window_start = 2w) (tested on RTFS, RGB and I420 pushes).
 * Crops are extracted lazily (only the canvas pixels inside each rectangle), so a caller with an exact CNN-feature
 * cache keyed by (target k, rect) can skip crops it already has (shv_participant_window_info + _crop); the same
 * (k, rect) always yields the same crop bytes.
 * Typical loop:  push(...) -> if res.ready_window >= 0 && res.ready_status == SHV_WIN_READY:
 *                    shv_participant_window(p, res.ready_window, crops, &info, targets, times)
 *                    shv_participant_release_before(p, res.ready_window + 1)   (optional: peak 20 instead of 23 frames)
 *                else if res.ready_status == SHV_WIN_UNAVAILABLE: report the window unavailable (NaN score).
 * Frames are released automatically when the next window resolves (the most recently resolved window stays
 * extractable until then).  Memory = cached frames x frame bytes + ~1 KB: shv_participant_bytes() reports it
 * (1080p, AUTO storage: 23 x 2.76 MB = 63.6 MB peak measured, 55 MB with the explicit release).
 * Errors: a failing detection on a sampled frame marks its target MISSING (windows containing it UNAVAILABLE).
 * Not thread-safe per participant (one call at a time); distinct participants may run on distinct threads.
 */
typedef struct shv_participant shv_participant;
#define SHV_WIN_COLLECTING 0   /* not all 20 targets resolved yet */
#define SHV_WIN_READY 1
#define SHV_WIN_UNAVAILABLE 2  /* a target of the window is MISSING (frame gap > 0.25 s) */
#define SHV_WIN_EXPIRED 3      /* its frames were already released */
typedef struct {
    int64_t selected_target;   /* target index this frame was sampled for, or -1 */
    int64_t missing_first, missing_count;   /* targets declared missing by this push */
    int64_t ready_window;      /* window index whose last target this push resolved (ready or unavailable), or -1 */
    int ready_status;          /* status of ready_window */
} shv_push_result;
/* cascade: shared, must outlive the participant.  start: first target time after the first frame (0 live). */
shv_participant* shv_participant_create(const shv_cascade* cascade, double start, int hop_targets);
void shv_participant_free(shv_participant* p);
/* Frame storage (exact in every mode):
 *   SHV_STORE_NATIVE: a copy of the pushed frame (RGB24 3 B/px or I420 1.5 B/px);
 *   SHV_STORE_SMALL : the reference's resized frame `small` = cv2.resize(native, fit 1280x720) as RGB24 (the canvas
 *                     without its black bars), computed once per sampled frame; crops then only run the final
 *                     INTER_AREA;
 *   SHV_STORE_AUTO (default): whichever is smaller for this frame (1080p RGB -> SMALL 2.76 MB instead of 6.2 MB;
 *                     1080p I420 -> SMALL 2.76 MB instead of 3.1 MB; 720p I420 -> NATIVE 1.38 MB). */
#define SHV_STORE_AUTO 0
#define SHV_STORE_NATIVE 1
#define SHV_STORE_SMALL 2
int shv_participant_set_store(shv_participant* p, int mode);
int shv_participant_push_rgb(shv_participant* p, double t, const shv_rgb_image* frame, shv_push_result* res);
int shv_participant_push_i420(shv_participant* p, double t, const shv_i420_image* frame, shv_push_result* res);
/* Mark the stream broken from target k on (decode failure): every window containing a target >= k that is not
 * resolved yet becomes UNAVAILABLE. */
void shv_participant_mark_missing(shv_participant* p, int64_t first_target, int64_t count);
int shv_participant_window_status(const shv_participant* p, int64_t w);
/* Geometry of a READY window (cheap: cached Haar results + interpolation). */
int shv_participant_window_info(const shv_participant* p, int64_t w, shv_window_info* info,
                                int64_t targets[20], double times[20]);
/* Lazy crop of sampled frame (target) k for any rect; k must still be cached. */
int shv_participant_crop(const shv_participant* p, int64_t k, shv_rect rect, uint8_t* out);
/* All 20 crops of a READY window (crops: 20x180x320x3).  info/targets/times may be NULL. */
int shv_participant_window(const shv_participant* p, int64_t w, uint8_t* crops, shv_window_info* info,
                           int64_t targets[20], double times[20]);
/* Release frames no window >= w needs (targets < hop*w); windows < w become EXPIRED. */
void shv_participant_release_before(shv_participant* p, int64_t w);
size_t shv_participant_bytes(const shv_participant* p);
int shv_participant_cached_frames(const shv_participant* p);

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
#ifdef __cplusplus
}
#endif
#endif /* SHV_PREP_H */
