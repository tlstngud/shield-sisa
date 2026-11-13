// verify_service_edge: edge-case / lifetime tests of libshield_av (complementing bench/service_contracts).
// Every emitted result of every test also passes a global invariant check (hop order, NaN-never-zero, -1 decisions,
// anchors only when valid, JSON null for NaN).
//
//   verify_service_edge [--backend cpu|cuda] [--only NAME] [--fast]
// Tests:
//   gap_boundary   frames exactly 0.25 s after a target (valid) vs 0.25 s + 1e-9 (unavailable/gap), expectation computed
//   cache_onoff    the same real clip with the exact token cache on and off: bitwise identical logits
//   reset_inflight reset while video/audio jobs are queued and in flight, then the same input == a fresh session
//   free_inflight  free a session with jobs in flight (lifetime; run under ASan)
//   slots          max_sessions 3 (CUDA token slot blocks): 4th create fails, free + create reuses the block, results ok
//   sizes          tiny / odd / large frames (RGB and I420): accepted or rejected cleanly, no crash, valid results
//   big_chunk      60 s of audio in one push after video: hops emitted in order, video lag handled
//   finish_timeout finish(0) with jobs in flight returns -1 (timeout), finish(-1) then completes and emits
//   json_escape    participant id with quotes / backslash / control characters -> valid JSON line (checked by python)
//   threads        16 sessions, audio and video pushed from separate threads concurrently (TSan target)
#include "data_path.hpp"
#include "service_common.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace svc;

namespace {
int g_fail = 0;
std::string g_backend = "cpu";
#define CHECK(c, ...)                                                   \
    do {                                                                \
        if (!(c)) {                                                     \
            ++g_fail;                                                   \
            std::fprintf(stderr, "  FAIL %s:%d: %s | ", __FILE__, __LINE__, #c); \
            std::fprintf(stderr, __VA_ARGS__);                          \
            std::fprintf(stderr, "\n");                                 \
        }                                                               \
    } while (0)

void check_part(const shav_modality_result& m, bool audio, int64_t hop) {
    if (m.status == SHAV_VALID) {
        CHECK(std::isfinite(m.logit) && std::isfinite(m.score), "hop %lld valid but NaN", (long long)hop);
        CHECK(m.decision == 0 || m.decision == 1, "hop %lld decision %d", (long long)hop, m.decision);
        CHECK(m.alert == 0 || m.alert == 1, "hop %lld alert %d", (long long)hop, m.alert);
        CHECK(std::fabs(m.score - 1.0 / (1.0 + std::exp(-double(m.logit)))) < 1e-12, "score != sigmoid(logit)");
        CHECK(m.decision == (m.score >= m.threshold ? 1 : 0), "decision rule");
        if (!audio) CHECK(m.face_detection_anchors >= 0 && m.face_detection_anchors <= 5, "anchors %d", m.face_detection_anchors);
    } else {
        CHECK(std::isnan(m.logit) && std::isnan(m.score), "hop %lld status %d reason %d logit %g score %g (must be NaN)",
              (long long)hop, m.status, m.reason, double(m.logit), m.score);
        CHECK(m.decision == -1 && m.alert == -1, "hop %lld non-valid decision %d alert %d", (long long)hop, m.decision, m.alert);
        if (!audio) CHECK(m.face_detection_anchors == -1, "anchors on non-valid hop %lld", (long long)hop);
    }
}

struct Collector {
    std::mutex mu;
    std::vector<shav_hop_result> r;
    std::string id;
};
void on_result(const shav_hop_result* r, void* u) {
    Collector* c = static_cast<Collector*>(u);
    std::lock_guard<std::mutex> lk(c->mu);
    c->r.push_back(*r);
}

void check_all(const std::vector<shav_hop_result>& v, const char* id) {
    for (size_t i = 0; i < v.size(); ++i) {
        CHECK(v[i].hop == int64_t(i), "hops not consecutive from 0: index %zu hop %lld", i, (long long)v[i].hop);
        check_part(v[i].audio, true, v[i].hop);
        check_part(v[i].video, false, v[i].hop);
        char buf[4096];
        const int n = shav_result_json(&v[i], id, buf, sizeof buf);
        CHECK(n > 0 && n < int(sizeof buf), "json length %d", n);
        if (v[i].video.status != SHAV_VALID) CHECK(std::strstr(buf, "\"logit\":null") != nullptr, "json NaN not null: %s", buf);
        CHECK(std::strstr(buf, "nan") == nullptr && std::strstr(buf, "inf") == nullptr, "json has nan/inf: %s", buf);
    }
}

shav_engine* make_engine(int max_sessions = 64, int audio_workers = 4) {
    shav_engine_config ec;
    shav_engine_default_config(&ec);
    ec.audio_weights = "build/audio/shield.weights";
    ec.video_weights = "build/shield_v.weights";
    ec.video_backend = g_backend.c_str();
    ec.max_sessions = max_sessions;
    ec.audio_workers = audio_workers;
    shav_engine* e = shav_engine_create(&ec);
    if (!e) std::fprintf(stderr, "engine: %s\n", shav_last_error());
    return e;
}

shav_session* make_session(shav_engine* e, Collector* c, bool audio, bool video, int pending = 64, int cache = 1) {
    shav_session_config sc;
    shav_session_default_config(&sc);
    sc.participant_id = c->id.c_str();
    sc.audio_enabled = audio;
    sc.video_enabled = video;
    sc.video_max_pending = pending;
    sc.video_cache = cache;
    sc.callback = on_result;
    sc.user = c;
    return shav_session_create(e, &sc);
}

Clip* g_clip = nullptr;
Npy16* g_audio = nullptr;

void push_clip(shav_session* s, double t0, double seconds, bool audio) {
    const Clip& c = *g_clip;
    size_t a = 0;
    std::vector<float> pcm;
    for (size_t i = 0; i < c.t.size() && c.t[i] < seconds; ++i) {
        if (audio) {   // audio up to this frame time, from win_v2 rows (row r covers [10r, 10r+10) s)
            const size_t want = size_t((t0 + c.t[i]) * 16000.0);
            pcm.clear();
            for (; a < want; ++a) pcm.push_back(float(g_audio->data[(a / 160000) % 50 * 160000 + a % 160000]) / 32768.0f);
            if (!pcm.empty()) shav_session_push_audio(s, pcm.data(), pcm.size());
        }
        CHECK(push_frame(s, t0 + c.t[i], c.frame(i), c.w, c.h, c.yuv) == 0, "push frame %zu: %s", i, shav_last_error());
    }
}

// ------------------------------------------------------------------------------------------------ tests
void t_gap_boundary(shav_engine* e) {
    for (int late_variant = 0; late_variant < 2; ++late_variant) {
        Collector c;
        c.id = "gap";
        shav_session* s = make_session(e, &c, false, true);
        const int K = 60;           // targets 0..59 (30 s), one frame exactly at every target ...
        const int kgap = 25;        // ... except target 25: frame at 12.5 + 0.25 (+1e-9)
        const double late = late_variant ? 0.25 + 1e-9 : 0.25;
        const Clip& cl = *g_clip;
        for (int k = 0; k < K; ++k) {
            double t = 0.5 * k;
            if (k == kgap) t += late;
            CHECK(push_frame(s, t, cl.frame(size_t(k) % cl.t.size()), cl.w, cl.h, cl.yuv) == 0, "push %d: %s", k, shav_last_error());
        }
        shav_session_finish(s, -1);
        std::vector<shav_hop_result> v;
        {
            std::lock_guard<std::mutex> lk(c.mu);
            v = c.r;
        }
        check_all(v, "gap");
        // window w = hop - 4 covers targets 4w..4w+19; windows 0..10 are complete (targets up to 59)
        int checked = 0;
        for (const auto& r : v) {
            const int64_t w = r.hop - 4;
            if (w < 0 || 4 * w + 19 > K - 1) continue;
            const bool contains = 4 * w <= kgap && kgap <= 4 * w + 19;
            const bool expect_gap = late_variant && contains;
            ++checked;
            if (expect_gap)
                CHECK(r.video.status == SHAV_UNAVAILABLE && r.video.reason == SHAV_R_GAP, "late %.10f hop %lld status %d reason %d",
                      late, (long long)r.hop, r.video.status, r.video.reason);
            else
                CHECK(r.video.status == SHAV_VALID, "late %.10f hop %lld status %d reason %d", late, (long long)r.hop,
                      r.video.status, r.video.reason);
        }
        CHECK(checked == 11, "checked %d windows", checked);
        std::fprintf(stderr, "  gap_boundary late=%.10f: %zu hops, %d complete windows checked\n", late, v.size(), checked);
        shav_session_free(s);
    }
}

std::vector<shav_hop_result> run_av(shav_engine* e, double seconds, int cache, bool audio = true) {
    Collector c;
    c.id = "av";
    shav_session* s = make_session(e, &c, audio, true, 64, cache);
    push_clip(s, 0.0, seconds, audio);
    shav_session_finish(s, -1);
    shav_session_free(s);
    check_all(c.r, "av");
    return c.r;
}

bool same_results(const std::vector<shav_hop_result>& a, const std::vector<shav_hop_result>& b, int* nvalid) {
    if (a.size() != b.size()) return false;
    *nvalid = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].video.status != b[i].video.status || a[i].audio.status != b[i].audio.status) return false;
        if (std::memcmp(&a[i].video.logit, &b[i].video.logit, 4) || std::memcmp(&a[i].audio.logit, &b[i].audio.logit, 4)) return false;
        if (a[i].video.face_detection_anchors != b[i].video.face_detection_anchors) return false;
        *nvalid += a[i].video.status == SHAV_VALID;
    }
    return true;
}

void t_cache_onoff(shav_engine* e) {
    auto on = run_av(e, 30.0, 1, false);
    auto off = run_av(e, 30.0, 0, false);
    int nv = 0;
    CHECK(same_results(on, off, &nv), "cache on/off results differ");
    int reused = 0;
    for (auto& r : on) reused += r.video.frames_reused > 0 ? r.video.frames_reused : 0;
    int reused_off = 0;
    for (auto& r : off) reused_off += r.video.frames_reused > 0 ? r.video.frames_reused : 0;
    std::fprintf(stderr, "  cache_onoff: %zu hops, %d valid video, frames reused on=%d off=%d\n", on.size(), nv, reused, reused_off);
    CHECK(reused_off == 0, "cache off but %d frames reused", reused_off);
}

void t_reset_inflight(shav_engine* e) {
    auto fresh = run_av(e, 24.0, 1);
    Collector c;
    c.id = "reset";
    shav_session* s = make_session(e, &c, true, true);
    for (int round = 0; round < 3; ++round) {
        push_clip(s, 0.0, 24.0, true);   // fast feed: jobs are queued / in flight now
        CHECK(shav_session_reset(s) == 0, "reset: %s", shav_last_error());
        std::lock_guard<std::mutex> lk(c.mu);
        c.r.clear();   // results delivered before the reset belong to the old timeline
    }
    push_clip(s, 0.0, 24.0, true);
    shav_session_finish(s, -1);
    std::vector<shav_hop_result> v;
    {
        std::lock_guard<std::mutex> lk(c.mu);
        v = c.r;
    }
    check_all(v, "reset");
    int nv = 0;
    CHECK(same_results(fresh, v, &nv), "after mid-flight resets: %zu vs %zu hops", fresh.size(), v.size());
    std::fprintf(stderr, "  reset_inflight: %zu hops, %d valid video, equal to a fresh session\n", v.size(), nv);
    shav_session_free(s);
}

void t_free_inflight(shav_engine* e) {
    for (int i = 0; i < 4; ++i) {
        Collector c;
        c.id = "free";
        shav_session* s = make_session(e, &c, true, true);
        push_clip(s, 0.0, 20.0 + 3 * i, true);
        shav_session_free(s);   // no finish: queued jobs dropped, in-flight awaited
    }
    shav_engine_stats st;
    shav_engine_get_stats(e, &st);
    CHECK(st.sessions == 0, "sessions %lld after free", (long long)st.sessions);
    std::fprintf(stderr, "  free_inflight: ok, crop buffers %zu\n", st.crop_buffers);
}

void t_slots() {
    shav_engine* e = make_engine(3);
    Collector c[5];
    shav_session* s[5] = {};
    for (int i = 0; i < 3; ++i) {
        c[i].id = "slot" + std::to_string(i);
        s[i] = make_session(e, &c[i], false, true);
        CHECK(s[i] != nullptr, "create %d: %s", i, shav_last_error());
    }
    c[3].id = "slot3";
    s[3] = make_session(e, &c[3], false, true);
    if (g_backend == "cuda") CHECK(s[3] == nullptr, "4th video session created with max_sessions 3");
    if (s[3]) shav_session_free(s[3]);
    shav_session_free(s[1]);
    c[4].id = "slot4";
    s[4] = make_session(e, &c[4], false, true);
    CHECK(s[4] != nullptr, "create after free: %s", shav_last_error());
    // all live sessions score the same clip identically
    for (int i : {0, 2, 4}) push_clip(s[i], 0.0, 16.0, false);
    for (int i : {0, 2, 4}) shav_session_finish(s[i], -1);
    int nv = 0;
    CHECK(same_results(c[0].r, c[4].r, &nv) && same_results(c[0].r, c[2].r, &nv), "slot reuse changes results");
    for (int i : {0, 2, 4}) {
        check_all(c[i].r, "slot");
        shav_session_free(s[i]);
    }
    std::fprintf(stderr, "  slots: %d valid video per session, identical across sessions / reused block\n", nv);
    shav_engine_free(e);
}

void t_sizes(shav_engine* e) {
    struct Sz {
        int w, h;
    } sizes[] = {{1, 1}, {2, 2}, {15, 9}, {33, 17}, {321, 181}, {641, 359}, {1919, 1081}, {3840, 2160}, {4000, 100}, {100, 4000}};
    for (const Sz& z : sizes)
        for (int yuv = 0; yuv < 2; ++yuv) {
            Collector c;
            c.id = "size";
            shav_session* s = make_session(e, &c, false, true);
            std::vector<uint8_t> rgb(size_t(z.w) * z.h * 3), y(size_t(z.w) * z.h), u(size_t((z.w + 1) / 2) * ((z.h + 1) / 2));
            for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t((i * 2654435761u) >> 24);
            for (size_t i = 0; i < y.size(); ++i) y[i] = uint8_t(16 + (i * 40503u >> 8) % 220);
            for (size_t i = 0; i < u.size(); ++i) u[i] = uint8_t(128 + int((i * 7) % 40) - 20);
            int rc_bad = 0;
            for (int k = 0; k < 24 * 2; ++k) {   // 24 s at 2 fps, exactly on the targets
                const double t = 0.5 * k;
                const int rc = yuv ? shav_session_push_video_i420(s, t, y.data(), z.w, u.data(), (z.w + 1) / 2, u.data(),
                                                                  (z.w + 1) / 2, z.w, z.h)
                                   : shav_session_push_video_rgb(s, t, rgb.data(), z.w, z.h, ptrdiff_t(3) * z.w);
                rc_bad += rc != 0;
            }
            shav_session_finish(s, -1);
            check_all(c.r, "size");
            int valid = 0;
            for (auto& r : c.r) valid += r.video.status == SHAV_VALID;
            std::fprintf(stderr, "  sizes %s %dx%d: push failures %d, hops %zu, valid %d\n", yuv ? "i420" : "rgb", z.w, z.h,
                         rc_bad, c.r.size(), valid);
            CHECK(rc_bad == 0 || rc_bad == 48, "partial push failures %d", rc_bad);
            if (rc_bad == 0) CHECK(valid == 8, "expected 8 valid windows (hops 4..11: targets 0..47), got %d", valid);
            shav_session_free(s);
        }
}

void t_big_chunk(shav_engine* e) {
    Collector c;
    c.id = "chunk";
    shav_session* s = make_session(e, &c, true, true);
    push_clip(s, 0.0, 30.0, false);
    std::vector<float> pcm(60 * 16000);
    for (size_t a = 0; a < pcm.size(); ++a) pcm[a] = float(g_audio->data[(a / 160000) % 50 * 160000 + a % 160000]) / 32768.0f;
    CHECK(shav_session_push_audio(s, pcm.data(), pcm.size()) == 0, "push 60 s: %s", shav_last_error());
    shav_session_finish(s, -1);
    check_all(c.r, "chunk");
    int lag = 0, valid_v = 0, valid_a = 0;
    for (auto& r : c.r) {
        lag += r.video.reason == SHAV_R_LAG;
        valid_v += r.video.status == SHAV_VALID;
        valid_a += r.audio.status == SHAV_VALID || r.audio.status == SHAV_INSUFFICIENT_SPEECH;
    }
    CHECK(c.r.size() == 30, "hops %zu (60 s of audio = 30 hops)", c.r.size());
    std::fprintf(stderr, "  big_chunk: %zu hops, video valid %d, video lag %d, audio scored %d\n", c.r.size(), valid_v, lag, valid_a);
    shav_session_free(s);
}

void t_finish_timeout(shav_engine* e) {
    Collector c;
    c.id = "timeout";
    shav_session* s = make_session(e, &c, true, true);
    push_clip(s, 0.0, 30.0, true);
    const int rc0 = shav_session_finish(s, 0.0);
    const int rc1 = shav_session_finish(s, -1);
    CHECK(rc1 == 0, "finish(-1) rc %d", rc1);
    check_all(c.r, "timeout");
    std::fprintf(stderr, "  finish_timeout: finish(0) rc %d (expected -1 while jobs in flight), finish(-1) rc %d, hops %zu\n", rc0,
                 rc1, c.r.size());
    for (auto& r : c.r)
        std::fprintf(stderr, "    hop %lld audio %s/%s video %s/%s\n", (long long)r.hop, shav_status_name(r.audio.status),
                     shav_reason_name(r.audio.reason), shav_status_name(r.video.status), shav_reason_name(r.video.reason));
    // 14 complete audio hops + hop 14 (audio unavailable/ended, video valid): video results are not lost at finish (F3)
    CHECK(c.r.size() == 15 && c.r.back().audio.status == SHAV_UNAVAILABLE && c.r.back().audio.reason == SHAV_R_ENDED, "hops %zu (expected 14 complete + 1 ended)", c.r.size());
    CHECK(shav_session_push_audio(s, nullptr, 0) != 0, "push after finish accepted");
    shav_session_free(s);
}

void t_json_escape() {
    shav_hop_result r;
    std::memset(&r, 0, sizeof r);
    r.audio.logit = r.video.logit = std::nanf("");
    r.audio.score = r.video.score = std::nan("");
    r.audio.decision = r.audio.alert = r.video.decision = r.video.alert = r.alert = -1;
    r.video.face_detection_anchors = r.video.frames_encoded = r.video.frames_reused = -1;
    r.audio.t_input = r.audio.t_done = r.video.t_input = r.video.t_done = std::nan("");
    const char* ids[] = {"plain", "q\"uote", "back\\slash", "tab\tnew\nline", "ctl\x01\x1f", "utf8 \xec\x8b\xa0"};
    const char* td = std::getenv("TMPDIR");
    const std::string out = std::string(td && *td ? td : "/tmp") + "/shield_edge_json.jsonl";
    FILE* f = std::fopen(out.c_str(), "w");
    CHECK(f != nullptr, "cannot write %s", out.c_str());
    if (!f) return;
    for (const char* id : ids) {
        char buf[4096];
        shav_result_json(&r, id, buf, sizeof buf);
        std::fprintf(f, "%s\n", buf);
    }
    std::fclose(f);
    std::fprintf(stderr, "  json_escape: wrote %s\n", out.c_str());
}

// resolution changes mid-stream (WebRTC adapts resolution): every 3rd frame is a 640x360 / odd 333x187 I420 frame
void t_res_change(shav_engine* e) {
    Collector c;
    c.id = "res";
    shav_session* s = make_session(e, &c, false, true);
    const Clip& cl = *g_clip;
    std::vector<uint8_t> y(640 * 360), u(320 * 180);
    for (size_t i = 0; i < y.size(); ++i) y[i] = uint8_t(16 + (i * 7) % 200);
    for (size_t i = 0; i < u.size(); ++i) u[i] = uint8_t(100 + i % 50);
    int fails = 0;
    for (size_t i = 0; i < cl.t.size() && cl.t[i] < 24.0; ++i) {
        int rc;
        if (i % 3 == 1) rc = shav_session_push_video_i420(s, cl.t[i], y.data(), 640, u.data(), 320, u.data(), 320, 640, 360);
        else if (i % 3 == 2) rc = shav_session_push_video_i420(s, cl.t[i], y.data(), 333, u.data(), 167, u.data(), 167, 333, 187);
        else rc = push_frame(s, cl.t[i], cl.frame(i), cl.w, cl.h, cl.yuv);
        fails += rc != 0;
    }
    shav_session_finish(s, -1);
    check_all(c.r, "res");
    int valid = 0;
    for (auto& r : c.r) valid += r.video.status == SHAV_VALID;
    std::fprintf(stderr, "  res_change: push failures %d, hops %zu, valid video %d\n", fails, c.r.size(), valid);
    CHECK(fails == 0 && valid == 8, "fails %d valid %d", fails, valid);
    shav_session_free(s);
}

// audio stops at 12 s while video continues to 30 s (muted / dropped microphone): are the video results of hops
// after 12 s ever emitted?  (The symmetric case, a stalled camera, is handled by the video lag rule.)
void t_audio_stall(shav_engine* e) {
    Collector c;
    c.id = "astall";
    shav_session* s = make_session(e, &c, true, true);
    const Clip& cl = *g_clip;
    std::vector<float> pcm;
    size_t a = 0;
    for (size_t i = 0; i < cl.t.size() && cl.t[i] < 30.0; ++i) {
        const size_t want = size_t(std::min(cl.t[i], 12.0) * 16000.0);
        pcm.clear();
        for (; a < want; ++a) pcm.push_back(float(g_audio->data[a]) / 32768.0f);
        if (!pcm.empty()) shav_session_push_audio(s, pcm.data(), pcm.size());
        push_frame(s, cl.t[i], cl.frame(i), cl.w, cl.h, cl.yuv);
    }
    shav_session_wait_idle(s, -1);
    size_t before;
    {
        std::lock_guard<std::mutex> lk(c.mu);
        before = c.r.size();
    }
    shav_session_stats st;
    shav_session_get_stats(s, &st);
    shav_session_finish(s, -1);
    check_all(c.r, "astall");
    std::fprintf(stderr, "  audio_stall: video windows scored %lld, hops emitted before finish %zu, after finish %zu "
                 "(video-complete hops up to 14 exist)\n", (long long)st.video_windows_scored, before, c.r.size());
    shav_session_free(s);
}

void t_threads(shav_engine* e) {
    const int N = 16;
    std::vector<Collector> c(N);
    std::vector<shav_session*> s(N);
    for (int i = 0; i < N; ++i) {
        c[size_t(i)].id = "thr" + std::to_string(i);
        s[size_t(i)] = make_session(e, &c[size_t(i)], true, true, 2);
    }
    std::vector<std::thread> th;
    std::vector<std::atomic<double>> vt(N);
    for (auto& x : vt) x.store(-1.0);
    for (int i = 0; i < N; ++i) {
        th.emplace_back([&, i] {   // video thread
            const Clip& cl = *g_clip;
            for (size_t k = 0; k < cl.t.size() && cl.t[k] < 20.0; ++k) {
                push_frame(s[size_t(i)], cl.t[k], cl.frame(k), cl.w, cl.h, cl.yuv);
                vt[size_t(i)].store(cl.t[k]);
            }
            vt[size_t(i)].store(1e9);
        });
        th.emplace_back([&, i] {   // audio thread, paced behind the video thread (lockstep like a real call)
            std::vector<float> pcm(3200);
            for (size_t a0 = 0; a0 < size_t(20 * 16000); a0 += pcm.size()) {
                while (vt[size_t(i)].load() < double(a0 + pcm.size()) / 16000.0) std::this_thread::yield();
                for (size_t j = 0; j < pcm.size(); ++j)
                    pcm[j] = float(g_audio->data[((a0 + j) / 160000) % 50 * 160000 + (a0 + j) % 160000]) / 32768.0f;
                shav_session_push_audio(s[size_t(i)], pcm.data(), pcm.size());
            }
        });
    }
    for (auto& t : th) t.join();
    int nv = 0, drops = 0;
    for (int i = 0; i < N; ++i) {
        shav_session_finish(s[size_t(i)], -1);
        check_all(c[size_t(i)].r, "thr");
        for (auto& r : c[size_t(i)].r) {
            nv += r.video.status == SHAV_VALID;
            drops += r.video.reason == SHAV_R_OVERLOAD || r.video.reason == SHAV_R_LAG;
        }
        shav_session_free(s[size_t(i)]);
    }
    std::fprintf(stderr, "  threads: %d sessions, valid video %d, overload/lag %d\n", N, nv, drops);
}
}  // namespace

int main(int argc, char** argv) {
    std::string only;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--backend") g_backend = argv[++i];
        else if (a == "--only") only = argv[++i];
    }
    auto rows = load_map("test");
    Clip clip;
    std::string err;
    const MapRow* row = nullptr;
    for (const auto& r : rows) {   // first test Fake whose name says it is >= 31 s long (..._SSSSS_EEEEE_...)
        if (r.label != 1) continue;
        int a = -1, b = -1;
        for (size_t p = 0; p + 11 < r.sample_id.size(); ++p)
            if (r.sample_id[p] == '_' && r.sample_id[p + 6] == '_' && std::isdigit((unsigned char)r.sample_id[p + 1]) &&
                std::isdigit((unsigned char)r.sample_id[p + 7]) && std::isdigit((unsigned char)r.sample_id[p + 11])) {
                a = std::atoi(r.sample_id.c_str() + p + 1);
                b = std::atoi(r.sample_id.c_str() + p + 7);
                break;
            }
        if (a >= 0 && b - a >= 31) { row = &r; break; }
    }
    if (!decode_clip(row->path, true, 31.0, clip, err)) {
        std::fprintf(stderr, "decode: %s\n", err.c_str());
        return 2;
    }
    g_clip = &clip;
    Npy16 au;
    if (!au.open(shvbench_data("/win_v2/test.npy"))) return 2;
    g_audio = &au;
    std::fprintf(stderr, "backend %s, clip %s %dx%d %zu frames\n", g_backend.c_str(), row->sample_id.c_str(), clip.w, clip.h,
                 clip.t.size());
    shav_engine* e = make_engine();
    if (!e) return 2;
    struct T {
        const char* name;
        std::function<void()> f;
    } tests[] = {
        {"gap_boundary", [&] { t_gap_boundary(e); }}, {"cache_onoff", [&] { t_cache_onoff(e); }},
        {"reset_inflight", [&] { t_reset_inflight(e); }}, {"free_inflight", [&] { t_free_inflight(e); }},
        {"slots", [&] { t_slots(); }}, {"sizes", [&] { t_sizes(e); }}, {"big_chunk", [&] { t_big_chunk(e); }},
        {"finish_timeout", [&] { t_finish_timeout(e); }}, {"json_escape", [&] { t_json_escape(); }},
        {"threads", [&] { t_threads(e); }}, {"audio_stall", [&] { t_audio_stall(e); }},
        {"res_change", [&] { t_res_change(e); }},
    };
    for (auto& t : tests) {
        if (!only.empty() && only != t.name) continue;
        const int before = g_fail;
        std::fprintf(stderr, "[%s]\n", t.name);
        t.f();
        std::fprintf(stderr, "[%s] %s\n", t.name, g_fail == before ? "PASS" : "FAIL");
    }
    shav_engine_free(e);
    std::fprintf(stderr, "TOTAL failures %d\n", g_fail);
    return g_fail ? 1 : 0;
}
