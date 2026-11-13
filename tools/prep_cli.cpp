// prep_cli: SHIELD-V native preprocessing of one video window, decoding exactly like reference/shim/av.py
// (ffprobe packet PTS -> sample_frames selection -> one ffmpeg call selecting those PTS, -noautorotate,
// -copyts, scale=in_color_matrix=bt601:in_range=auto:flags=bilinear,format=rgb24).
//
// usage: prep_cli VIDEO WINDOW_START [--out FILE|-] [--yuv] [--isa scalar|avx2|avx512] [--cascade XML]
//                 [--repeat N]
//   --out    write the 20x180x320x3 uint8 crops (raw, C order) to FILE ('-' = stdout)
//   --yuv    decode yuv420p instead and use the native I420->RGB converter (approximate tier check)
//   --repeat run the preprocessing N times (timing)
// stdout (unless --out -): one JSON line {anchors, frame_ids, times, rects, faces, timing_ms{...}}
// exit status 0 ok, 2 usage, 3 sampling/decoding failure (JSON line with "error").
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "shv_prep.h"

#include "prep_decode.hpp"

static void fail(const char* msg) {
    printf("{\"error\": \"%s\"}\n", msg);
    exit(3);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: prep_cli VIDEO WINDOW_START [--out FILE|-] [--yuv] [--isa L] [--cascade XML] [--repeat N]\n");
        return 2;
    }
    std::string path = argv[1];
    double start = strtod(argv[2], nullptr);
    const char* out = nullptr;
    const char* cascade_path = nullptr;
    bool yuv = false;
    int repeat = 1;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--yuv")) yuv = true;
        else if (!strcmp(argv[i], "--cascade") && i + 1 < argc) cascade_path = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--isa") && i + 1 < argc) {
            const char* l = argv[++i];
            int lv = !strcmp(l, "scalar") ? SHV_ISA_SCALAR : !strcmp(l, "avx2") ? SHV_ISA_AVX2
                                                           : !strcmp(l, "avx512") ? SHV_ISA_AVX512 : SHV_ISA_AUTO;
            if (shv_prep_set_isa(lv) < 0) fail("isa not supported");
        } else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    double t0 = now_ms();
    Probe pr;
    std::string err;
    if (!probe(path, pr, err)) fail(err.c_str());
    double t_probe = now_ms();
    double src_dur = NAN;
    if (pr.has_duration_ts) src_dur = shv_pts_seconds(pr.duration_ts, pr.tb_num, pr.tb_den);
    else if (pr.has_container) src_dur = (double)pr.container_us / 1000000.0;
    else fail("Video duration is unavailable");
    size_t idx[20];
    double times[20];
    int rc = shv_sample_window(pr.pts.data(), pr.pts.size(), pr.tb_num, pr.tb_den, start, src_dur, idx, times);
    if (rc < 0) fail(shv_prep_last_error());
    std::vector<int64_t> want(20);
    for (int i = 0; i < 20; i++) want[i] = pr.pts[idx[i]];
    std::string raw;
    if (!decode(path, want, yuv, pr.width, pr.height, raw, err)) fail(err.c_str());
    double t_dec = now_ms();
    shv_cascade* cas = shv_cascade_load(cascade_path);
    if (!cas) fail(shv_prep_last_error());
    std::vector<uint8_t> crops((size_t)20 * SHV_CROP_BYTES);
    shv_window_info info;
    double best = 1e30, total = 0;
    const int w = pr.width, h = pr.height;
    for (int r = 0; r < repeat; r++) {
        double a = now_ms();
        if (!yuv) {
            shv_rgb_image fr[20];
            for (int i = 0; i < 20; i++)
                fr[i] = shv_rgb_image{(const uint8_t*)raw.data() + (size_t)i * w * h * 3, w, h, (ptrdiff_t)w * 3};
            rc = shv_window_prepare_rgb(cas, fr, crops.data(), &info);
        } else {
            shv_i420_image fr[20];
            int cw = (w + 1) / 2, ch = (h + 1) / 2;
            size_t fb = (size_t)w * h + 2 * (size_t)cw * ch;
            for (int i = 0; i < 20; i++) {
                const uint8_t* Y = (const uint8_t*)raw.data() + i * fb;
                fr[i] = shv_i420_image{Y, Y + (size_t)w * h, Y + (size_t)w * h + (size_t)cw * ch, w, cw, cw, w, h};
            }
            rc = shv_window_prepare_i420(cas, fr, crops.data(), &info);
        }
        if (rc < 0) fail(shv_prep_last_error());
        double d = now_ms() - a;
        best = std::min(best, d);
        total += d;
    }
    shv_cascade_free(cas);
    if (out) {
        FILE* f = !strcmp(out, "-") ? stdout : fopen(out, "wb");
        if (!f) fail("cannot open output");
        fwrite(crops.data(), 1, crops.size(), f);
        if (f != stdout) fclose(f);
    }
    if (!out || strcmp(out, "-")) {
        printf("{\"anchors\": %d, \"width\": %d, \"height\": %d, \"isa\": \"%s\", \"frame_ids\": [", info.anchors, w, h,
               shv_prep_isa_name(shv_prep_isa()));
        for (int i = 0; i < 20; i++) printf("%s%lld", i ? ", " : "", (long long)want[i]);
        printf("], \"times\": [");
        for (int i = 0; i < 20; i++) printf("%s%.17g", i ? ", " : "", times[i]);
        printf("], \"rects\": [");
        for (int i = 0; i < 20; i++)
            printf("%s[%d, %d, %d, %d]", i ? ", " : "", info.rects[i].x, info.rects[i].y, info.rects[i].w,
                   info.rects[i].h);
        printf("], \"faces\": [");
        for (int i = 0; i < 5; i++) {
            if (info.faces[i].found)
                printf("%s[%d, %d, %d, %d]", i ? ", " : "", info.faces[i].box.x, info.faces[i].box.y,
                       info.faces[i].box.w, info.faces[i].box.h);
            else
                printf("%snull", i ? ", " : "");
        }
        printf("], \"timing_ms\": {\"probe\": %.3f, \"decode\": %.3f, \"prep_best\": %.3f, \"prep_mean\": %.3f}}\n",
               t_probe - t0, t_dec - t_probe, best, total / repeat);
    }
    return 0;
}
