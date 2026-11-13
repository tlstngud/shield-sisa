// Runtime ISA selection and thread-local error strings.
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "internal.hpp"

namespace shvp {

static thread_local char g_err[512] = "";

void set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

bool cpu_has_avx2() {
#if defined(__x86_64__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("bmi") &&
           __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt");
#else
    return false;
#endif
}

bool cpu_has_avx512() {
#if defined(__x86_64__)
    __builtin_cpu_init();
    return cpu_has_avx2() && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
#else
    return false;
#endif
}

bool cpu_has_vbmi() {
#if defined(__x86_64__)
    __builtin_cpu_init();
    return cpu_has_avx512() && __builtin_cpu_supports("avx512vbmi");
#else
    return false;
#endif
}

static int best_isa() { return cpu_has_avx512() ? SHV_ISA_AVX512 : cpu_has_avx2() ? SHV_ISA_AVX2 : SHV_ISA_SCALAR; }

static std::atomic<int> g_isa{0};

static int env_isa() {
    const char* e = getenv("SHV_PREP_ISA");
    if (!e || !*e) return 0;
    if (!strcmp(e, "scalar")) return SHV_ISA_SCALAR;
    if (!strcmp(e, "avx2")) return SHV_ISA_AVX2;
    if (!strcmp(e, "avx512")) return SHV_ISA_AVX512;
    return 0;
}

int isa() {
    int v = g_isa.load(std::memory_order_relaxed);
    if (v) return v;
    int want = env_isa();
    int best = best_isa();
    v = (want && want <= best) ? want : best;
    g_isa.store(v, std::memory_order_relaxed);
    return v;
}

}  // namespace shvp

extern "C" {

int shv_prep_isa_supported(int level) {
    using namespace shvp;
    switch (level) {
        case SHV_ISA_SCALAR: return 1;
        case SHV_ISA_AVX2: return cpu_has_avx2() ? 1 : 0;
        case SHV_ISA_AVX512: return cpu_has_avx512() ? 1 : 0;
        default: return 0;
    }
}

int shv_prep_set_isa(int level) {
    using namespace shvp;
    if (level == SHV_ISA_AUTO) {
        g_isa.store(0);
        return isa();
    }
    if (!shv_prep_isa_supported(level)) {
        set_error("ISA level %d not supported by this CPU", level);
        return SHV_EINVAL;
    }
    g_isa.store(level);
    return level;
}

int shv_prep_isa(void) { return shvp::isa(); }

const char* shv_prep_isa_name(int level) {
    switch (level) {
        case SHV_ISA_SCALAR: return "scalar";
        case SHV_ISA_AVX2: return "avx2";
        case SHV_ISA_AVX512: return "avx512";
        default: return "auto";
    }
}

const char* shv_prep_last_error(void) { return shvp::g_err; }

const char* shv_prep_version(void) {
    return "shield_v_prep 1 (Pillow 11.3 BILINEAR, OpenCV 4.13 cvtColor/resize/CascadeClassifier, numpy 2.0, "
           "ffmpeg 6.1 swscale yuv2rgb)";
}
}
