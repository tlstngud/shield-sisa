// prep_bench: per-stage timing of libshield_v_prep on real RTFS windows, per dispatch level, single thread.
// usage: prep_bench MAP.jsonl [--limit N] [--stride K] [--repeat R] [--isa scalar,avx2,avx512] [--yuv]
// Decodes the 20 sampled frames of each selected video once (prep_decode.hpp, reference shim path), then times:
//   letterbox (per frame), gray+haar (per anchor frame), lazy crop (per frame), whole window (20 frames),
//   and with --yuv the I420 variants.  Prints one JSON object with medians (ms) and a crop checksum per level
//   (must be identical across levels).
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "prep_decode.hpp"
#include "shv_prep.h"

struct Video {
    std::string path;
    double start;
    int w, h;
    std::string rgb, yuv;
};

static double med(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static uint64_t fnv(const uint8_t* d, size_t n, uint64_t h = 1469598103934665603ULL) {
    for (size_t i = 0; i < n; i++) h = (h ^ d[i]) * 1099511628211ULL;
    return h;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: prep_bench MAP.jsonl [--limit N] [--stride K] [--repeat R] [--isa list] [--yuv]\n");
        return 2;
    }
    int limit = 8, stride = 1, repeat = 5;
    bool yuv = false;
    std::string isas = "scalar,avx2,avx512";
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--limit")) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stride")) stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat")) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--isa")) isas = argv[++i];
        else if (!strcmp(argv[i], "--yuv")) yuv = true;
    }
    FILE* f = fopen(argv[1], "r");
    if (!f) return 1;
    std::vector<Video> vids;
    char line[8192];
    int ln = 0;
    while (fgets(line, sizeof line, f) && (int)vids.size() < limit) {
        if ((ln++ % stride) != 0) continue;
        std::string s(line);
        auto get = [&](const char* k) {
            size_t p = s.find(std::string("\"") + k + "\": ");
            if (p == std::string::npos) return std::string();
            p += strlen(k) + 4;
            if (s[p] == '"') {
                size_t e = s.find('"', p + 1);
                return s.substr(p + 1, e - p - 1);
            }
            size_t e = s.find_first_of(",}", p);
            return s.substr(p, e - p);
        };
        Video v;
        v.path = get("path");
        v.start = strtod(get("window_start").c_str(), nullptr);
        Probe pr;
        std::string err;
        if (!probe(v.path, pr, err)) continue;
        double dur = pr.has_duration_ts ? shv_pts_seconds(pr.duration_ts, pr.tb_num, pr.tb_den)
                                        : (double)pr.container_us / 1e6;
        size_t idx[20];
        double times[20];
        if (shv_sample_window(pr.pts.data(), pr.pts.size(), pr.tb_num, pr.tb_den, v.start, dur, idx, times) < 0)
            continue;
        std::vector<int64_t> want(20);
        for (int i = 0; i < 20; i++) want[i] = pr.pts[idx[i]];
        if (!decode(v.path, want, false, pr.width, pr.height, v.rgb, err)) continue;
        if (yuv && !decode(v.path, want, true, pr.width, pr.height, v.yuv, err)) continue;
        v.w = pr.width, v.h = pr.height;
        vids.push_back(std::move(v));
    }
    fclose(f);
    shv_cascade* cas = shv_cascade_load(nullptr);
    std::vector<uint8_t> lb(SHV_CROP_BYTES), crop(SHV_CROP_BYTES), crops((size_t)20 * SHV_CROP_BYTES);
    printf("{\"videos\": %zu, \"repeat\": %d, \"levels\": {", vids.size(), repeat);
    bool first_level = true;
    size_t pos = 0;
    while (pos <= isas.size()) {
        size_t e = isas.find(',', pos);
        if (e == std::string::npos) e = isas.size();
        std::string lv = isas.substr(pos, e - pos);
        pos = e + 1;
        int L = lv == "scalar" ? SHV_ISA_SCALAR : lv == "avx2" ? SHV_ISA_AVX2 : SHV_ISA_AVX512;
        if (shv_prep_set_isa(L) < 0) continue;
        std::map<std::string, std::vector<double>> T;
        uint64_t sum_crops = 1469598103934665603ULL;
        for (auto& v : vids) {
            size_t fbytes = (size_t)v.w * v.h * 3;
            shv_rgb_image fr[20];
            for (int i = 0; i < 20; i++)
                fr[i] = shv_rgb_image{(const uint8_t*)v.rgb.data() + i * fbytes, v.w, v.h, (ptrdiff_t)v.w * 3};
            shv_window_info info;
            for (int r = 0; r < repeat; r++) {
                double a = now_ms();
                for (int i = 0; i < 20; i++) shv_letterbox_rgb(&fr[i], lb.data());
                T["letterbox_per_frame"].push_back((now_ms() - a) / 20);
                const int anc[5] = {0, 5, 10, 15, 19};
                double hsum = 0;
                for (int k = 0; k < 5; k++) {
                    shv_letterbox_rgb(&fr[anc[k]], lb.data());
                    shv_face face;
                    double b = now_ms();
                    shv_detect_face_letterboxed(cas, lb.data(), &face);
                    hsum += now_ms() - b;
                }
                T["gray_haar_per_frame"].push_back(hsum / 5);
                a = now_ms();
                shv_window_prepare_rgb(cas, fr, crops.data(), &info);
                T["window_rgb"].push_back(now_ms() - a);
                a = now_ms();
                for (int i = 0; i < 20; i++) shv_native_crop_rgb(&fr[i], info.rects[i], crop.data());
                T["crop_per_frame"].push_back((now_ms() - a) / 20);
            }
            sum_crops = fnv(crops.data(), crops.size(), sum_crops);
            if (yuv) {
                int cw = (v.w + 1) / 2, ch = (v.h + 1) / 2;
                size_t yb = (size_t)v.w * v.h + 2 * (size_t)cw * ch;
                shv_i420_image fy[20];
                for (int i = 0; i < 20; i++) {
                    const uint8_t* Y = (const uint8_t*)v.yuv.data() + i * yb;
                    fy[i] = shv_i420_image{Y, Y + (size_t)v.w * v.h, Y + (size_t)v.w * v.h + (size_t)cw * ch, v.w, cw, cw,
                                           v.w, v.h};
                }
                std::vector<uint8_t> full(fbytes), crops2((size_t)20 * SHV_CROP_BYTES);
                for (int r = 0; r < repeat; r++) {
                    double a = now_ms();
                    for (int i = 0; i < 20; i++) shv_i420_to_rgb24(&fy[i], full.data(), (ptrdiff_t)v.w * 3);
                    T["i420_to_rgb_full_per_frame"].push_back((now_ms() - a) / 20);
                    a = now_ms();
                    for (int i = 0; i < 20; i++) shv_letterbox_i420(&fy[i], lb.data());
                    T["letterbox_i420_per_frame"].push_back((now_ms() - a) / 20);
                    a = now_ms();
                    for (int i = 0; i < 20; i++) shv_native_crop_i420(&fy[i], info.rects[i], crop.data());
                    T["crop_i420_per_frame"].push_back((now_ms() - a) / 20);
                    a = now_ms();
                    shv_window_prepare_i420(cas, fy, crops2.data(), nullptr);
                    T["window_i420"].push_back(now_ms() - a);
                }
                if (memcmp(crops2.data(), crops.data(), crops.size())) fprintf(stderr, "i420 crops differ: %s\n", v.path.c_str());
            }
        }
        printf("%s\"%s\": {\"crop_checksum\": \"%016llx\"", first_level ? "" : ", ", lv.c_str(),
               (unsigned long long)sum_crops);
        for (auto& kv : T) printf(", \"%s\": %.4f", kv.first.c_str(), med(kv.second));
        printf("}");
        first_level = false;
        fflush(stdout);
    }
    printf("}}\n");
    shv_cascade_free(cas);
    return 0;
}
