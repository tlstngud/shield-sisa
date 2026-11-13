// service_contracts: status contract, timeline, cache-exactness, policy and independence tests of libshield_av.
//
//   service_contracts --backend cuda|cpu [--out bench/artifacts/service/contracts_<backend>.json]
//
// Inputs: two RTFS test videos decoded into memory (I420 1080p, looped with continuing timestamps), win_v2 test audio.
// Feeding is lockstep on the session clock (audio up to t, then every video frame with time < t), 20 ms steps.
// Tests (each prints one JSON line; exit status 1 if any fails):
//   short_input   8.5 s: hops 0..3 collecting (audio warmup, video warmup), nothing more after finish
//   steady        30 s: hops >= 4 valid in both modalities; every video logit == the engine's batch predict on the
//                 crops the window used (crop tap), bitwise, although later windows reuse cached tokens
//   frame_gap     frames in [11.99, 13.0) dropped: exactly the hops whose window contains a target with no frame within
//                 0.25 s are unavailable (gap), computed independently from the frame list
//   reset         24 s, reset, same 24 s again == the first run == a fresh session (bitwise)
//   no_face       uniform grey frames: valid with face_detection_anchors 0 and whole-frame rectangles
//   video_stall   video stops at 16 s, audio continues to 30 s: later hops unavailable (lag), last one ended
//   no_video      audio only into an AV session: video collecting (no_input) on every hop, audio valid
//   silence       zero PCM: audio insufficient_speech, video valid, combined alert = video alert
//   invalid_input NaN / out-of-range audio and non-increasing video timestamps are rejected without state change
//   policies      the combined alert follows the documented truth table for all four policies
//   video_only    audio-disabled session: hops emitted from the video clock, audio unavailable (disabled)
//   late_video    first frame at 5.3 s: hops 4..6 video collecting (warmup), hop 7 onwards valid
//   backpressure  video_max_pending 0, fast feed: drops are unavailable/overload (NaN), audio never dropped, every
//                 scored window == batch predict on its crops, drop count == engine stats
//   independence  participant A alone vs A among 31 concurrent participants: identical results (bitwise)
// The status tests use video_max_pending 64: input is fed faster than real time, so a slow backend (CPU) would
// otherwise drop windows by design.
#include "data_path.hpp"
#include "service_common.hpp"

#include <atomic>
#include <functional>
#include <mutex>
#include <thread>

using namespace svc;

namespace {

shav_engine* g_engine = nullptr;
std::vector<Clip> g_clips;
Npy16 g_audio;
int g_fail = 0;
std::string g_json;

void report(const std::string& name, bool ok, const std::string& detail) {
    std::printf("{\"test\":\"%s\",\"pass\":%s,%s}\n", name.c_str(), ok ? "true" : "false", detail.c_str());
    std::fflush(stdout);
    g_fail += !ok;
    g_json += std::string(g_json.empty() ? "" : ",") + "{\"test\":\"" + name + "\",\"pass\":" + (ok ? "true" : "false") + "," + detail + "}";
}

struct Collector {
    std::mutex mu;
    std::vector<shav_hop_result> r;
};
void collect_cb(const shav_hop_result* r, void* u) {
    Collector* c = static_cast<Collector*>(u);
    std::lock_guard<std::mutex> lk(c->mu);
    c->r.push_back(*r);
}
struct TapStore {
    std::mutex mu;
    std::map<int64_t, std::vector<uint8_t>> crops;
    std::map<int64_t, std::vector<std::array<int, 4>>> rects;
};
void tap_cb(const shav_window_view* v, void* u) {
    TapStore* t = static_cast<TapStore*>(u);
    std::lock_guard<std::mutex> lk(t->mu);
    t->crops[v->hop].assign(v->crops, v->crops + 20 * size_t(172800));
    auto& rr = t->rects[v->hop];
    rr.clear();
    for (int i = 0; i < 20; ++i) rr.push_back({v->rects[i][0], v->rects[i][1], v->rects[i][2], v->rects[i][3]});
}

struct Opts {
    int clip = 0;
    size_t audio_row = 0;
    bool silence = false, no_video = false, no_audio = false;
    double drop_from = -1, drop_to = -1;   // video frames with t in [from, to) are not pushed
    double video_stop = 1e30;               // no video frames at/after this time
    double video_offset = 0;                // added to every frame time
    bool grey = false;
};

// video frame n of the looped clip (time, pointer)
struct VideoSrc {
    const Clip* c;
    double period;
    std::vector<uint8_t> grey;
    explicit VideoSrc(const Clip* c_) : c(c_) {
        const double dt = c->t.size() > 1 ? c->t[1] - c->t[0] : 1.0 / 30;
        period = c->t.back() + dt;
    }
    double time(size_t n) const { return double(n / c->t.size()) * period + c->t[n % c->t.size()]; }
    const uint8_t* frame(size_t n) const { return c->frame(n % c->t.size()); }
};

struct Feeder {
    shav_session* s;
    Opts o;
    VideoSrc v;
    size_t next_frame = 0, audio_pos = 0;
    std::vector<float> chunk;
    std::vector<uint8_t> grey;
    Feeder(shav_session* s_, const Opts& o_) : s(s_), o(o_), v(&g_clips[size_t(o_.clip)]) {
        if (o.grey) {
            const Clip& c = g_clips[size_t(o.clip)];
            grey.assign(c.frame_bytes, 128);
        }
    }
    int16_t sample(size_t i) const {
        if (o.silence) return 0;
        const size_t row = (o.audio_row + i / 160000) % g_audio.rows;
        return g_audio.row(row)[i % 160000];
    }
    // push everything up to session time T
    bool until(double T) {
        const size_t A = size_t(std::llround(T * 16000));
        if (!o.no_audio && A > audio_pos) {
            chunk.resize(A - audio_pos);
            for (size_t i = 0; i < chunk.size(); ++i) chunk[i] = float(sample(audio_pos + i)) / 32768.0f;
            if (shav_session_push_audio(s, chunk.data(), chunk.size()) != 0) return false;
            audio_pos = A;
        }
        if (o.no_video) return true;
        for (;;) {
            const double t = v.time(next_frame) + o.video_offset;
            if (t >= T) break;
            const size_t n = next_frame++;
            if (t >= o.video_stop) continue;
            if (t >= o.drop_from && t < o.drop_to) continue;
            const Clip& c = *v.c;
            if (push_frame(s, t, o.grey ? grey.data() : v.frame(n), c.w, c.h, true) != 0) return false;
        }
        return true;
    }
    bool run(double T) {
        for (double t = 0.02; t <= T + 1e-9; t += 0.02)
            if (!until(t)) return false;
        return true;
    }
};

shav_session_config base_cfg(const char* id, Collector* col, TapStore* tap) {
    shav_session_config c;
    shav_session_default_config(&c);
    c.participant_id = id;
    c.callback = collect_cb;
    c.user = col;
    // inputs are fed faster than real time: a deep pending queue keeps the status tests independent of backend speed
    // (backpressure has its own test below)
    c.video_max_pending = 64;
    if (tap) {
        c.crop_tap = tap_cb;
        c.crop_tap_user = tap;
    }
    return c;
}

std::vector<shav_hop_result> run_session(const shav_session_config& cfg, const Opts& o, double T, bool finish = true,
                                         Collector* col_out = nullptr) {
    Collector* col = static_cast<Collector*>(cfg.user);
    shav_session* s = shav_session_create(g_engine, &cfg);
    if (!s) {
        std::fprintf(stderr, "session: %s\n", shav_last_error());
        std::exit(1);
    }
    Feeder f(s, o);
    if (!f.run(T)) std::fprintf(stderr, "feed failed: %s\n", shav_last_error());
    if (finish) shav_session_finish(s, 120.0);
    else shav_session_wait_idle(s, 120.0);
    shav_session_free(s);
    (void)col_out;
    std::lock_guard<std::mutex> lk(col->mu);
    return col->r;
}

bool same_part(const shav_modality_result& a, const shav_modality_result& b) {
    return a.status == b.status && a.reason == b.reason && std::memcmp(&a.logit, &b.logit, 4) == 0 &&
           (a.score == b.score || (std::isnan(a.score) && std::isnan(b.score))) && a.decision == b.decision &&
           a.alert == b.alert && a.face_detection_anchors == b.face_detection_anchors && a.frames_encoded == b.frames_encoded &&
           a.frames_reused == b.frames_reused;
}
bool same_results(const std::vector<shav_hop_result>& a, const std::vector<shav_hop_result>& b, std::string* why = nullptr) {
    if (a.size() != b.size()) {
        if (why) *why = "count " + std::to_string(a.size()) + " vs " + std::to_string(b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].hop != b[i].hop || a[i].alert != b[i].alert || !same_part(a[i].audio, b[i].audio) || !same_part(a[i].video, b[i].video)) {
            if (why) *why = "hop " + std::to_string(a[i].hop);
            return false;
        }
    return true;
}
std::string statuses(const std::vector<shav_hop_result>& r) {
    std::string s = "\"hops\":[";
    for (size_t i = 0; i < r.size(); ++i) {
        s += (i ? "," : "") + std::string("[") + std::to_string(r[i].hop) + ",\"" + shav_status_name(r[i].audio.status) + "/" +
             shav_reason_name(r[i].audio.reason) + "\",\"" + shav_status_name(r[i].video.status) + "/" +
             shav_reason_name(r[i].video.reason) + "\"," + (r[i].alert < 0 ? "null" : std::to_string(r[i].alert)) + "]";
    }
    return s + "]";
}
bool in_order(const std::vector<shav_hop_result>& r) {
    for (size_t i = 0; i < r.size(); ++i)
        if (r[i].hop != int64_t(i)) return false;
    return true;
}
bool nan_contract(const shav_modality_result& m) {
    if (m.status == SHAV_VALID) return std::isfinite(m.logit) && std::isfinite(m.score) && m.decision >= 0 && m.alert >= 0;
    return std::isnan(m.logit) && std::isnan(m.score) && m.decision == -1 && m.alert == -1;
}
int expected_alert(int policy, const shav_hop_result& r) {
    const bool av = r.audio.status == SHAV_VALID, vv = r.video.status == SHAV_VALID;
    switch (policy) {
        case SHAV_POLICY_AUDIO: return av ? r.audio.alert : -1;
        case SHAV_POLICY_VIDEO: return vv ? r.video.alert : -1;
        case SHAV_POLICY_BOTH:
            if (av && vv) return r.audio.alert == 1 && r.video.alert == 1;
            if ((av && r.audio.alert == 0) || (vv && r.video.alert == 0)) return 0;
            return -1;
        default:
            if ((av && r.audio.alert == 1) || (vv && r.video.alert == 1)) return 1;
            return (av || vv) ? 0 : -1;
    }
}

// sampler rule on a frame list: is any target of hop h's window missing?
bool window_has_gap(const std::vector<double>& ft, int64_t h) {
    for (int i = 0; i < 20; ++i) {
        const double tau = 2.0 * double(h) - 8.0 + i * 0.5;
        auto it = std::lower_bound(ft.begin(), ft.end(), tau - 1e-6);
        if (it == ft.end() || *it - tau > 0.25) return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------ tests
void t_short() {
    Collector col;
    auto cfg = base_cfg("short", &col, nullptr);
    Opts o;
    auto r = run_session(cfg, o, 8.5);
    bool ok = r.size() == 4 && in_order(r);
    for (auto& h : r)
        ok &= h.audio.status == SHAV_COLLECTING && h.audio.reason == SHAV_R_WARMUP && h.video.status == SHAV_COLLECTING &&
              h.video.reason == SHAV_R_WARMUP && h.alert == -1 && nan_contract(h.audio) && nan_contract(h.video);
    report("short_input", ok, statuses(r));
}

void t_steady() {
    Collector col;
    TapStore tap;
    auto cfg = base_cfg("steady", &col, &tap);
    Opts o;
    auto r = run_session(cfg, o, 30.0);
    bool ok = r.size() == 15 && in_order(r);
    int valid_v = 0, valid_a = 0, bitwise = 0, reused = 0, checked = 0;
    for (auto& h : r) {
        ok &= nan_contract(h.audio) && nan_contract(h.video);
        if (h.hop < 4) { ok &= h.audio.status == SHAV_COLLECTING && h.video.status == SHAV_COLLECTING; continue; }
        valid_a += h.audio.status == SHAV_VALID;
        ok &= h.audio.status == SHAV_VALID || h.audio.status == SHAV_INSUFFICIENT_SPEECH;
        ok &= h.video.status == SHAV_VALID && h.video.face_detection_anchors >= 0 && h.video.face_detection_anchors <= 5;
        ok &= std::fabs(h.video.window_start - (2.0 * h.hop - 8)) < 1e-12 && h.audio.window_end == 2.0 * h.hop + 2;
        valid_v += h.video.status == SHAV_VALID;
        reused += h.video.frames_reused;
        ok &= h.video.frames_encoded + h.video.frames_reused == 20;
        ok &= h.video.score == 1.0 / (1.0 + std::exp(-double(h.video.logit)));
        // live (cached tokens) == batch predict on the window's crops
        auto it = tap.crops.find(h.hop);
        if (it != tap.crops.end()) {
            float lg = NAN;
            shav_engine_video_predict(g_engine, it->second.data(), 1, &lg);
            ++checked;
            bitwise += std::memcmp(&lg, &h.video.logit, 4) == 0;
        }
    }
    ok &= checked == valid_v && bitwise == checked && valid_v == 11;
    report("steady", ok, "\"valid_audio\":" + std::to_string(valid_a) + ",\"valid_video\":" + std::to_string(valid_v) +
                             ",\"live_eq_batch\":" + std::to_string(bitwise) + ",\"live_checked\":" + std::to_string(checked) +
                             ",\"frames_reused\":" + std::to_string(reused) + "," + statuses(r));
}

void t_gap() {
    Collector col;
    auto cfg = base_cfg("gap", &col, nullptr);
    Opts o;
    o.drop_from = 11.99;
    o.drop_to = 13.0;
    auto r = run_session(cfg, o, 30.0);
    // independent expectation from the pushed frame times
    VideoSrc v(&g_clips[0]);
    std::vector<double> ft;
    for (size_t n = 0;; ++n) {
        const double t = v.time(n);
        if (t >= 30.0) break;
        if (t >= o.drop_from && t < o.drop_to) continue;
        ft.push_back(t);
    }
    bool ok = r.size() == 15 && in_order(r);
    int gaps = 0;
    std::string exp = "\"expected_gap_hops\":[";
    for (auto& h : r) {
        if (h.hop < 4) continue;
        const bool g = window_has_gap(ft, h.hop);
        if (g) exp += (gaps++ ? "," : "") + std::to_string(h.hop);
        if (g) ok &= h.video.status == SHAV_UNAVAILABLE && h.video.reason == SHAV_R_GAP && nan_contract(h.video);
        else ok &= h.video.status == SHAV_VALID;
    }
    ok &= gaps == 5;
    report("frame_gap", ok, exp + "]," + statuses(r));
}

void t_reset() {
    Collector c1, c2;
    auto cfg = base_cfg("reset", &c1, nullptr);
    shav_session* s = shav_session_create(g_engine, &cfg);
    Opts o;
    o.clip = 1;
    o.audio_row = 40;
    {
        Feeder f(s, o);
        f.run(24.0);
    }
    shav_session_finish(s, 120);
    std::vector<shav_hop_result> first;
    {
        std::lock_guard<std::mutex> lk(c1.mu);
        first = c1.r;
        c1.r.clear();
    }
    shav_session_reset(s);
    {
        Feeder f(s, o);
        f.run(24.0);
    }
    shav_session_finish(s, 120);
    shav_session_free(s);
    std::vector<shav_hop_result> second = c1.r;
    auto cfg2 = base_cfg("reset", &c2, nullptr);
    auto fresh = run_session(cfg2, o, 24.0);
    std::string w1, w2;
    const bool a = same_results(first, second, &w1), b = same_results(first, fresh, &w2);
    int valid = 0;
    for (auto& h : first) valid += h.video.status == SHAV_VALID;
    report("reset", a && b && first.size() == 12 && valid == 8,
           "\"after_reset_equal\":" + std::string(a ? "true" : "false") + ",\"fresh_equal\":" + (b ? "true" : "false") +
               ",\"why\":\"" + w1 + w2 + "\",\"hops\":" + std::to_string(first.size()));
}

void t_noface() {
    Collector col;
    TapStore tap;
    auto cfg = base_cfg("noface", &col, &tap);
    Opts o;
    o.grey = true;
    auto r = run_session(cfg, o, 14.0);
    bool ok = r.size() == 7;
    int n = 0;
    for (auto& h : r) {
        if (h.hop < 4) continue;
        ok &= h.video.status == SHAV_VALID && h.video.face_detection_anchors == 0;
        for (auto& rc : tap.rects[h.hop]) ok &= rc[0] == 0 && rc[1] == 0 && rc[2] == 320 && rc[3] == 180;
        ++n;
    }
    report("no_face", ok && n == 3, statuses(r));
}

void t_stall() {
    Collector col;
    auto cfg = base_cfg("stall", &col, nullptr);
    Opts o;
    o.video_stop = 16.0;
    auto r = run_session(cfg, o, 30.0);
    bool ok = r.size() == 15 && in_order(r);
    for (auto& h : r) {
        if (h.hop < 4) continue;
        if (2.0 * h.hop + 1.5 < 16.0 - 0.04) ok &= h.video.status == SHAV_VALID;
        else if (h.hop <= 13) ok &= h.video.status == SHAV_UNAVAILABLE && h.video.reason == SHAV_R_LAG;
        else ok &= h.video.status == SHAV_UNAVAILABLE && h.video.reason == SHAV_R_ENDED;
        ok &= h.audio.status == SHAV_VALID || h.audio.status == SHAV_INSUFFICIENT_SPEECH;
    }
    report("video_stall", ok, statuses(r));
}

void t_novideo() {
    Collector col;
    auto cfg = base_cfg("novideo", &col, nullptr);
    Opts o;
    o.no_video = true;
    auto r = run_session(cfg, o, 14.0);
    bool ok = r.size() == 7 && in_order(r);
    for (auto& h : r) {
        ok &= h.video.status == SHAV_COLLECTING && h.video.reason == SHAV_R_NO_INPUT;
        if (h.hop >= 4) ok &= h.audio.status == SHAV_VALID && h.alert == h.audio.alert;
    }
    report("no_video", ok, statuses(r));
}

void t_silence() {
    Collector col;
    auto cfg = base_cfg("silence", &col, nullptr);
    Opts o;
    o.silence = true;
    auto r = run_session(cfg, o, 14.0);
    bool ok = r.size() == 7;
    for (auto& h : r) {
        if (h.hop < 4) continue;
        ok &= h.audio.status == SHAV_INSUFFICIENT_SPEECH && h.audio.reason == SHAV_R_SILENCE && nan_contract(h.audio);
        ok &= h.video.status == SHAV_VALID && h.alert == h.video.alert;
    }
    report("silence", ok, statuses(r));
}

void t_invalid() {
    Collector col;
    auto cfg = base_cfg("invalid", &col, nullptr);
    shav_session* s = shav_session_create(g_engine, &cfg);
    std::vector<float> a(32000, 0.01f);
    a[100] = NAN;
    bool ok = shav_session_push_audio(s, a.data(), a.size()) == -1;
    a[100] = 1.5f;
    ok &= shav_session_push_audio(s, a.data(), a.size()) == -1;
    shav_session_stats st;
    shav_session_get_stats(s, &st);
    ok &= st.audio_hops == 0;
    a[100] = 0.0f;
    ok &= shav_session_push_audio(s, a.data(), a.size()) == 0;
    shav_session_get_stats(s, &st);
    ok &= st.audio_hops == 1;
    const Clip& c = g_clips[0];
    ok &= push_frame(s, 1.0, c.frame(0), c.w, c.h, true) == 0;
    ok &= push_frame(s, 1.0, c.frame(1), c.w, c.h, true) == -1;
    ok &= push_frame(s, 0.5, c.frame(1), c.w, c.h, true) == -1;
    ok &= push_frame(s, NAN, c.frame(1), c.w, c.h, true) == -1;
    ok &= shav_session_push_video_rgb(s, 2.0, nullptr, 10, 10, 30) == -1;
    ok &= push_frame(s, 1.04, c.frame(1), c.w, c.h, true) == 0;
    shav_session_get_stats(s, &st);
    ok &= st.video_frames_pushed == 2;
    shav_session_finish(s, 60);
    ok &= shav_session_push_audio(s, a.data(), 10) == -1;   // finished
    shav_session_free(s);
    report("invalid_input", ok, "\"detail\":\"NaN/out-of-range PCM, repeated/decreasing/NaN timestamps, null frame, push after finish\"");
}

void t_policies() {
    bool ok = true;
    std::string d = "\"checked\":[";
    for (int pol = 0; pol < 4; ++pol)
        for (double thr : {0.0, 1.0}) {
            Collector col;
            auto cfg = base_cfg("policy", &col, nullptr);
            cfg.policy = pol;
            cfg.audio_threshold = thr;   // 0: audio always alerts when valid, 1: (almost) never
            Opts o;
            o.drop_from = 15.9;
            o.drop_to = 17.0;   // some hops with video unavailable
            auto r = run_session(cfg, o, 26.0);
            int n = 0;
            for (auto& h : r) {
                ok &= h.alert == expected_alert(pol, h);
                ok &= h.policy == pol;
                if (h.audio.status == SHAV_VALID) ok &= h.audio.decision == (h.audio.score >= thr ? 1 : 0);
                ++n;
            }
            d += std::string(pol || thr > 0 ? "," : "") + "[" + std::to_string(pol) + "," + fmt(thr, "%.0f") + "," + std::to_string(n) + "]";
        }
    report("policies", ok, d + "]");
}

void t_videoonly() {
    Collector col;
    auto cfg = base_cfg("videoonly", &col, nullptr);
    cfg.audio_enabled = 0;
    Opts o;
    o.no_audio = true;
    auto r = run_session(cfg, o, 20.0);
    bool ok = r.size() >= 9 && in_order(r);
    for (auto& h : r) {
        ok &= h.audio.status == SHAV_UNAVAILABLE && h.audio.reason == SHAV_R_DISABLED;
        if (h.hop >= 4) ok &= h.video.status == SHAV_VALID && h.alert == h.video.alert;
    }
    report("video_only", ok, statuses(r));
}

void t_late() {
    Collector col;
    auto cfg = base_cfg("late", &col, nullptr);
    Opts o;
    o.video_offset = 5.3;
    auto r = run_session(cfg, o, 22.0);
    bool ok = r.size() == 11;
    for (auto& h : r) {
        if (h.hop < 4) continue;
        if (h.hop < 7) ok &= h.video.status == SHAV_COLLECTING && h.video.reason == SHAV_R_WARMUP;
        else ok &= h.video.status == SHAV_VALID;
    }
    report("late_video", ok, statuses(r));
}

// backpressure: video_max_pending = 0 and input fed as fast as possible; a ready window that finds the previous one
// still in flight is dropped (unavailable/overload, NaN), audio is never dropped, every scored window still equals
// batch predict on its own crops (the token cache stays exact across drops), and the drop count matches the engine
void t_backpressure() {
    Collector col;
    TapStore tap;
    auto cfg = base_cfg("backpressure", &col, &tap);
    cfg.video_max_pending = 0;
    shav_engine_stats e0, e1;
    shav_engine_get_stats(g_engine, &e0);
    Opts o;
    auto r = run_session(cfg, o, 40.0);
    shav_engine_get_stats(g_engine, &e1);
    bool ok = r.size() == 20 && in_order(r);
    int valid = 0, dropped = 0, checked = 0, bitwise = 0;
    for (auto& h : r) {
        ok &= nan_contract(h.audio) && nan_contract(h.video);
        if (h.hop < 4) continue;
        ok &= h.audio.status == SHAV_VALID || h.audio.status == SHAV_INSUFFICIENT_SPEECH;
        if (h.video.status == SHAV_VALID) {
            ++valid;
            auto it = tap.crops.find(h.hop);
            if (it != tap.crops.end()) {
                float lg = NAN;
                shav_engine_video_predict(g_engine, it->second.data(), 1, &lg);
                ++checked;
                bitwise += std::memcmp(&lg, &h.video.logit, 4) == 0;
            }
        } else {
            ok &= h.video.status == SHAV_UNAVAILABLE && h.video.reason == SHAV_R_OVERLOAD;
            ++dropped;
        }
    }
    ok &= valid > 0 && checked == valid && bitwise == checked && dropped == int(e1.video_dropped - e0.video_dropped);
    report("backpressure", ok, "\"valid_video\":" + std::to_string(valid) + ",\"dropped_overload\":" + std::to_string(dropped) +
                                   ",\"engine_dropped\":" + std::to_string(e1.video_dropped - e0.video_dropped) +
                                   ",\"live_eq_batch\":" + std::to_string(bitwise) + ",\"live_checked\":" + std::to_string(checked) +
                                   "," + statuses(r));
}

// participant A alone vs A among others (concurrent threads; GPU batches mix participants)
void t_independence() {
    const int P = 32;
    const double T = 30.0;
    auto run_group = [&](int n, std::vector<std::vector<shav_hop_result>>& out) {
        std::vector<Collector> cols(static_cast<size_t>(n));
        std::vector<std::string> ids(static_cast<size_t>(n));
        std::vector<shav_session*> ss(static_cast<size_t>(n));
        for (int p = 0; p < n; ++p) {
            ids[size_t(p)] = "p" + std::to_string(p);
            auto cfg = base_cfg(ids[size_t(p)].c_str(), &cols[size_t(p)], nullptr);
            cfg.video_max_pending = 64;
            ss[size_t(p)] = shav_session_create(g_engine, &cfg);
        }
        std::vector<std::thread> th;
        for (int p = 0; p < n; ++p)
            th.emplace_back([&, p] {
                Opts o;
                o.clip = p % int(g_clips.size());
                o.audio_row = size_t(p) * 7;
                o.video_offset = p == 0 ? 0.0 : 0.013 * p;   // participant 0 = A: identical input in both runs
                Feeder f(ss[size_t(p)], o);
                for (double t = 2.0; t <= T + 1e-9; t += 2.0) {
                    for (double u = t - 1.98; u <= t + 1e-9; u += 0.02) f.until(u);
                    shav_session_wait_idle(ss[size_t(p)], 120);
                }
                shav_session_finish(ss[size_t(p)], 120);
            });
        for (auto& t : th) t.join();
        out.resize(size_t(n));
        for (int p = 0; p < n; ++p) {
            shav_session_free(ss[size_t(p)]);
            out[size_t(p)] = cols[size_t(p)].r;
        }
    };
    std::vector<std::vector<shav_hop_result>> alone, together;
    run_group(1, alone);
    run_group(P, together);
    std::string why;
    const bool same = same_results(alone[0], together[0], &why);
    int vb = 0, maxb = 0;
    for (auto& h : together[0]) vb += h.video.status == SHAV_VALID;
    for (auto& g : together)
        for (auto& h : g) maxb = std::max(maxb, h.video.batch_windows);
    report("independence", same && vb == 11,
           "\"participants\":" + std::to_string(P) + ",\"identical\":" + (same ? "true" : "false") + ",\"why\":\"" + why +
               "\",\"valid_video_hops_A\":" + std::to_string(vb) + ",\"max_gpu_batch_windows_seen\":" + std::to_string(maxb));
}

}  // namespace

int main(int argc, char** argv) {
    std::string backend = "cuda", out;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
    }
    if (out.empty()) out = "bench/artifacts/service/contracts_" + backend + ".json";
    shav_engine_config c;
    shav_engine_default_config(&c);
    c.audio_weights = "build/audio/shield.weights";
    c.audio_workers = 4;
    c.video_weights = "build/shield_v.weights";
    c.video_backend = backend.c_str();
    c.max_sessions = 64;
    c.cpu_video_workers = 6;
    g_engine = shav_engine_create(&c);
    if (!g_engine) { std::fprintf(stderr, "engine: %s\n", shav_last_error()); return 1; }
    shav_engine_info info;
    shav_engine_get_info(g_engine, &info);
    std::printf("{\"event\":\"engine\",\"video_backend\":\"%s\",\"video_mode\":\"%s\",\"audio_mode\":\"%s\",\"load_log\":\"%s\"}\n",
                info.video_backend, info.video_mode, info.audio_mode, info.load_log);
    if (!g_audio.open(shvbench_data("/win_v2/test.npy"))) { std::fprintf(stderr, "audio npy\n"); return 1; }
    // two test videos (a Real and a Fake), I420 in memory
    auto rows = load_map("test");
    std::vector<std::string> pick;
    for (auto& r : rows)
        if (r.label == 0 && pick.empty()) pick.push_back(r.path);
    for (auto& r : rows)
        if (r.label == 1 && pick.size() == 1) pick.push_back(r.path);
    for (auto& p : pick) {
        Clip cl;
        std::string err;
        if (!decode_clip(p, true, 1e9, cl, err)) { std::fprintf(stderr, "decode %s: %s\n", p.c_str(), err.c_str()); return 1; }
        std::printf("{\"event\":\"clip\",\"path\":\"%s\",\"frames\":%zu,\"seconds\":%.3f,\"size\":\"%dx%d\"}\n", p.c_str(), cl.t.size(),
                    cl.t.back(), cl.w, cl.h);
        g_clips.push_back(std::move(cl));
    }
    std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"short_input", t_short}, {"steady", t_steady}, {"frame_gap", t_gap}, {"reset", t_reset},
        {"no_face", t_noface}, {"video_stall", t_stall}, {"no_video", t_novideo}, {"silence", t_silence},
        {"invalid_input", t_invalid}, {"policies", t_policies}, {"video_only", t_videoonly}, {"late_video", t_late},
        {"backpressure", t_backpressure}, {"independence", t_independence}};
    for (auto& t : tests)
        if (only.empty() || only == t.first) t.second();
    FILE* f = std::fopen(out.c_str(), "w");
    if (f) {
        std::fprintf(f, "{\"backend\":\"%s\",\"video_mode\":\"%s\",\"failed\":%d,\"tests\":[%s]}\n", info.video_backend, info.video_mode,
                     g_fail, g_json.c_str());
        std::fclose(f);
    }
    std::printf("{\"event\":\"done\",\"failed\":%d}\n", g_fail);
    shav_engine_free(g_engine);
    return g_fail ? 1 : 0;
}
