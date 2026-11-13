// SHIELD-V CPU engine benchmark / CLI (owned by the cpu_model agent).
//
//   cpu_bench --lib build/cpu/libshield_v_cpu.so [--lib2 OTHER.so] --weights build/shield_v.weights
//             --clips FILE.u8 [--threads 1] [--mode exact] [--isa auto] [--what LIST] [--reps 30] [--warmup 3]
//             [--logits OUT.txt]
// FILE.u8: raw uint8 clips, each 20 x 180 x 320 x 3 (bench/cpu_dump_clips.py writes them).
// --what: comma list of
//   frame1      encode 1 frame
//   frames4     encode 4 frames (live update, new frames only)
//   frames20    encode 20 frames
//   classify1   classify 1 window (tokens ready)
//   live4       encode 4 new frames + classify (exact token cache hit for the other 16)
//   live20      encode 20 frames + classify (crop rectangles changed: predict one clip)
//   classifyB   classify --batch windows;  predictB  predict --batch clips
// With --lib2 every repetition runs lib then lib2 (alternating order each repetition): paired comparison.
// Prints one JSON object per (lib, what) with latency stats in ms.
#include <dlfcn.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Counting allocator: the executable's definitions interpose malloc & co. for the dlopen'ed library too, so the
// benchmark can report how many heap allocations the engine makes during the timed (steady-state) repetitions.
#include <atomic>
extern "C" {
void* __libc_malloc(size_t);
void* __libc_calloc(size_t, size_t);
void* __libc_realloc(void*, size_t);
void* __libc_memalign(size_t, size_t);
void __libc_free(void*);
}
static std::atomic<long> g_allocs{0};
extern "C" void* malloc(size_t n) { g_allocs.fetch_add(1, std::memory_order_relaxed); return __libc_malloc(n); }
extern "C" void* calloc(size_t a, size_t b) { g_allocs.fetch_add(1, std::memory_order_relaxed); return __libc_calloc(a, b); }
extern "C" void* realloc(void* p, size_t n) { g_allocs.fetch_add(1, std::memory_order_relaxed); return __libc_realloc(p, n); }
extern "C" void free(void* p) { __libc_free(p); }
extern "C" int posix_memalign(void** p, size_t a, size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    *p = __libc_memalign(a, n);
    return *p ? 0 : 12;
}
extern "C" void* aligned_alloc(size_t a, size_t n) { g_allocs.fetch_add(1, std::memory_order_relaxed); return __libc_memalign(a, n); }
extern "C" void* memalign(size_t a, size_t n) { g_allocs.fetch_add(1, std::memory_order_relaxed); return __libc_memalign(a, n); }

struct Api {
    void* h = nullptr;
    std::string path;
    void* (*load_ex)(const char*, const char*, int, const char*);
    void* (*load_exp)(const char*, const char*, int, const char*);
    void (*free_)(void*);
    int (*encode)(void*, const uint8_t*, size_t, float*);
    int (*classify)(void*, const float*, size_t, float*);
    int (*classify_ptrs)(void*, const float* const*, size_t, float*);
    int (*predict)(void*, const uint8_t*, size_t, float*);
    const char* (*err)(void);
    const char* (*isa)(void*);
    void (*mem)(void*, size_t*, size_t*, size_t*);
    int (*prof)(void*, double*, long*, int);
    void* m = nullptr;
};

template <class F> void sym(Api& a, F& f, const char* name) {
    f = reinterpret_cast<F>(dlsym(a.h, name));
    if (!f) { std::fprintf(stderr, "missing symbol %s in %s\n", name, a.path.c_str()); std::exit(2); }
}
Api open_api(const std::string& path) {
    Api a;
    a.path = path;
    a.h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!a.h) { std::fprintf(stderr, "dlopen %s: %s\n", path.c_str(), dlerror()); std::exit(2); }
    sym(a, a.load_ex, "shv_cpu_load_ex");
    sym(a, a.load_exp, "shv_cpu_load_experimental");
    sym(a, a.free_, "shv_cpu_free");
    sym(a, a.encode, "shv_cpu_encode_frames");
    sym(a, a.classify, "shv_cpu_classify");
    sym(a, a.classify_ptrs, "shv_cpu_classify_ptrs");
    sym(a, a.predict, "shv_cpu_predict");
    sym(a, a.err, "shv_cpu_last_error");
    sym(a, a.isa, "shv_cpu_isa");
    sym(a, a.mem, "shv_cpu_memory");
    a.prof = reinterpret_cast<int (*)(void*, double*, long*, int)>(dlsym(a.h, "shv_cpu_profile_get"));
    if (!a.prof) a.prof = [](void*, double*, long*, int) { return 0; };
    return a;
}

double now_ms() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}
double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const double r = p / 100.0 * (v.size() - 1);
    const size_t i = (size_t)r;
    return i + 1 < v.size() ? v[i] + (r - i) * (v[i + 1] - v[i]) : v.back();
}

int main(int argc, char** argv) {
    std::string lib = "build/cpu/libshield_v_cpu.so", lib2, weights = "build/shield_v.weights", clips, mode = "exact",
                isa = "", what = "frame1,classify1,live4,live20", logits_out;
    int threads = 1, reps = 30, warmup = 3, batch = 8;
    bool experimental = false, profile = false;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (k == "--lib") lib = next();
        else if (k == "--lib2") lib2 = next();
        else if (k == "--weights") weights = next();
        else if (k == "--clips") clips = next();
        else if (k == "--threads") threads = std::atoi(next().c_str());
        else if (k == "--mode") mode = next();
        else if (k == "--isa") isa = next();
        else if (k == "--what") what = next();
        else if (k == "--reps") reps = std::atoi(next().c_str());
        else if (k == "--warmup") warmup = std::atoi(next().c_str());
        else if (k == "--batch") batch = std::atoi(next().c_str());
        else if (k == "--logits") logits_out = next();
        else if (k == "--experimental") experimental = true;
        else if (k == "--profile") profile = true;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (clips.empty()) { std::fprintf(stderr, "--clips required\n"); return 2; }
    std::ifstream in(clips, std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", clips.c_str()); return 2; }
    const size_t clip_bytes = (size_t)20 * 180 * 320 * 3;
    const size_t total = (size_t)in.tellg();
    if (total % clip_bytes || !total) { std::fprintf(stderr, "bad clip file size\n"); return 2; }
    const size_t nclips = total / clip_bytes;
    std::vector<uint8_t> data(total);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(data.data()), total);

    std::vector<Api> apis;
    apis.push_back(open_api(lib));
    if (!lib2.empty()) apis.push_back(open_api(lib2));
    for (Api& a : apis) {
        a.m = experimental ? a.load_exp(weights.c_str(), mode.c_str(), threads, isa.c_str())
                           : a.load_ex(weights.c_str(), mode.c_str(), threads, isa.c_str());
        if (!a.m) { std::fprintf(stderr, "load %s: %s\n", a.path.c_str(), a.err()); return 1; }
    }
    if (!logits_out.empty()) {  // score every clip (batch 1) with the first library
        std::FILE* f = std::fopen(logits_out.c_str(), "w");
        for (size_t c = 0; c < nclips; ++c) {
            float lg = NAN;
            if (apis[0].predict(apis[0].m, data.data() + c * clip_bytes, 1, &lg)) {
                std::fprintf(stderr, "predict: %s\n", apis[0].err());
                return 1;
            }
            std::fprintf(f, "%.9g\n", lg);
        }
        std::fclose(f);
    }
    const size_t maxb = std::max<size_t>(batch, 1);
    std::vector<float> tokens((size_t)std::max<size_t>(20, maxb * 20) * 16 * 384), lg(maxb);
    std::vector<uint8_t> bclips(maxb * clip_bytes);
    for (size_t b = 0; b < maxb; ++b)
        std::memcpy(bclips.data() + b * clip_bytes, data.data() + (b % nclips) * clip_bytes, clip_bytes);
    // live: cached tokens for frames 4..19 of the previous window, 4 new frames
    std::vector<const float*> ptrs(20);
    std::vector<std::string> items;
    {
        size_t s = 0;
        while (s <= what.size()) {
            size_t e = what.find(',', s);
            if (e == std::string::npos) e = what.size();
            if (e > s) items.push_back(what.substr(s, e - s));
            s = e + 1;
        }
    }
    for (const std::string& w : items) {
        std::vector<std::vector<double>> times(apis.size());
        auto run = [&](Api& a) -> double {
            const uint8_t* clip = data.data();
            const double t0 = now_ms();
            int rc = 0;
            if (w == "frame1") rc = a.encode(a.m, clip, 1, tokens.data());
            else if (w == "frames4") rc = a.encode(a.m, clip, 4, tokens.data());
            else if (w == "frames20") rc = a.encode(a.m, clip, 20, tokens.data());
            else if (w == "classify1") rc = a.classify(a.m, tokens.data(), 1, lg.data());
            else if (w == "live4") {
                rc = a.encode(a.m, clip + (size_t)16 * 180 * 320 * 3, 4, tokens.data() + (size_t)16 * 6144);
                if (!rc) rc = a.classify(a.m, tokens.data(), 1, lg.data());
            } else if (w == "live20") rc = a.predict(a.m, clip, 1, lg.data());
            else if (w == "classifyB") rc = a.classify(a.m, tokens.data(), maxb, lg.data());
            else if (w == "predictB") rc = a.predict(a.m, bclips.data(), maxb, lg.data());
            else { std::fprintf(stderr, "unknown --what %s\n", w.c_str()); std::exit(2); }
            const double t1 = now_ms();
            if (rc) { std::fprintf(stderr, "%s: %s\n", w.c_str(), a.err()); std::exit(1); }
            return t1 - t0;
        };
        // tokens must be valid for classify: encode one window first
        for (Api& a : apis) {
            if (a.encode(a.m, data.data(), 20, tokens.data())) { std::fprintf(stderr, "%s\n", a.err()); return 1; }
            for (size_t b = 1; b < maxb; ++b)
                std::memcpy(tokens.data() + b * 20 * 6144, tokens.data(), sizeof(float) * 20 * 6144);
        }
        for (int r = 0; r < warmup; ++r)
            for (Api& a : apis) run(a);
        if (profile) {  // reset counters
            std::vector<double> ns(64);
            std::vector<long> cn(64);
            for (Api& a : apis) a.prof(a.m, ns.data(), cn.data(), 64);
        }
        for (auto& t : times) t.reserve(reps);
        const long allocs0 = g_allocs.load();
        for (int r = 0; r < reps; ++r) {
            if (apis.size() == 2 && (r & 1)) {
                times[1].push_back(run(apis[1]));
                times[0].push_back(run(apis[0]));
            } else {
                for (size_t i = 0; i < apis.size(); ++i) times[i].push_back(run(apis[i]));
            }
        }
        const long steady_allocs = g_allocs.load() - allocs0;
        for (size_t i = 0; i < apis.size(); ++i) {
            const auto& t = times[i];
            double mean = 0;
            for (double x : t) mean += x;
            mean /= t.size();
            size_t wb = 0, ws = 0, sb = 0;
            apis[i].mem(apis[i].m, &wb, &ws, &sb);
            std::printf("{\"lib\": \"%s\", \"what\": \"%s\", \"threads\": %d, \"isa\": \"%s\", \"mode\": \"%s\", "
                        "\"batch\": %d, \"n\": %zu, \"p50_ms\": %.4f, \"p95_ms\": %.4f, \"p99_ms\": %.4f, "
                        "\"mean_ms\": %.4f, \"min_ms\": %.4f, \"max_ms\": %.4f, \"weight_bytes\": %zu, "
                        "\"workspace_bytes\": %zu, \"slot_bytes\": %zu, \"steady_state_allocs\": %ld, "
                        "\"process_allocs_total\": %ld}\n",
                        apis[i].path.c_str(), w.c_str(), threads, apis[i].isa(apis[i].m), mode.c_str(),
                        (w == "classifyB" || w == "predictB") ? batch : 1, t.size(), pct(t, 50), pct(t, 95),
                        pct(t, 99), mean, *std::min_element(t.begin(), t.end()),
                        *std::max_element(t.begin(), t.end()), wb, ws, sb, steady_allocs, g_allocs.load());
            std::fflush(stdout);
        }
        if (profile) {  // per-stage mean ms per repetition (rank 0 wall time)
            for (size_t i = 0; i < apis.size(); ++i) {
                std::vector<double> ns(64);
                std::vector<long> cn(64);
                if (!apis[i].prof(apis[i].m, ns.data(), cn.data(), 64)) continue;
                std::printf("{\"profile\": \"%s\", \"what\": \"%s\", \"stages_ms_per_rep\": {", apis[i].path.c_str(),
                            w.c_str());
                bool first = true;
                for (int s = 0; s < 64; ++s)
                    if (cn[s]) {
                        std::printf("%s\"%d\": %.4f", first ? "" : ", ", s, ns[s] / 1e6 / reps);
                        first = false;
                    }
                std::printf("}}\n");
            }
        }
        if (apis.size() == 2) {  // paired ratio per repetition
            std::vector<double> ratio;
            for (size_t r = 0; r < times[0].size(); ++r) ratio.push_back(times[1][r] / times[0][r]);
            std::printf("{\"paired\": \"%s\", \"what\": \"%s\", \"ratio_lib2_over_lib_p50\": %.4f, \"p05\": %.4f, "
                        "\"p95\": %.4f}\n",
                        apis[1].path.c_str(), w.c_str(), pct(ratio, 50), pct(ratio, 5), pct(ratio, 95));
        }
    }
    for (Api& a : apis) a.free_(a.m);
    return 0;
}
