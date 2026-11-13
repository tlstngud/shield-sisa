// service_load: real-time load benchmark of the SHIELD-AV service core (libshield_av).
//
//   service_load --participants N [--arrival burst|staggered] [--seconds T] [--backend cuda|cpu] [--video-mode M]
//                [--feeders F] [--audio-feeders FA] [--audio-workers A] [--cpu-video-workers W] [--blocking-sync 0|1] [--max-delay-us D]
//                [--clips K] [--clip-seconds S] [--no-video] [--no-audio] [--video-cache 0|1] [--out PREFIX]
//
// N synthetic participants.  Audio: win_v2 test windows (int16 / 32768) concatenated per participant.  Video: K RTFS
// test videos decoded once (ffmpeg, native yuv420p, presentation timestamps) into memory and looped with continuing
// timestamps (participant p plays clip p % K with a frame-phase offset < 1 frame).  F feeder threads push, at real-time
// rate on each participant's session clock, every due audio sample (10 ms granularity) and every due video frame
// (I420, all ~30 fps frames); they play the role of the decoder threads, so the service's video preprocessing runs on
// them (--audio-feeders FA > 0: FA extra threads push the audio instead, the way a server receives the two modalities
// on different threads; default 0 = the same thread pushes both, so video preprocessing lag delays the audio).  Arrivals: burst = every session clock starts at the same instant (all hop boundaries and video sampling
// instants coincide); staggered = participant p starts 2 p / N s later (hop boundaries spread over the 2 s period).
//
// Latency per hop h (measured hops 4 .. T/2-1, i.e. every hop with a full 10 s window):
//   audio : t_done - due, due = session start + 2(h+1) (wall time of the hop's last sample)
//   video : t_done - due, due = session start + time of the frame that completes the window (first frame at/after
//           target 2h+1.5) -- includes feeder lag, preprocessing, queueing and GPU/CPU inference
//   hop   : t_emit - (session start + 2(h+1)) (combined result available)
// Deadline miss: a hop emitted more than 2 s after its end, never emitted, or with a modality part that is unavailable
// because of lag / overload / failure.  CPU: process CPU seconds and per-thread-name CPU seconds over the measurement
// window [start + 10 s + stagger, start + T]; GPU: nvidia-smi samples (500 ms) in that window; memory: RSS before
// sessions / at the end of the window, peak RSS, shav_session_bytes.
#include "data_path.hpp"
#include "service_common.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

using namespace svc;

namespace {

struct Part {
    int id = 0;
    std::string pid;
    shav_session* s = nullptr;
    double s0 = 0;             // monotonic seconds of session time 0
    const Clip* clip = nullptr;
    double period = 0, voff = 0;
    size_t row0 = 0, apos = 0, vnext = 0;
    bool done = false, done_a = false;   // done: video/both feeder finished; done_a: audio feeder finished
    std::mutex mu;
    std::vector<shav_hop_result> res;
    double vt(size_t n) const {
        const size_t nf = clip->t.size();
        return voff + double(n / nf) * period + clip->t[n % nf];
    }
    // session time of the first frame at/after tau (sampler rule, 1e-6 slack)
    double frame_at(double tau) const {
        size_t lo = 0, hi = 1;
        while (vt(hi) < tau - 1e-6) hi *= 2;
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (vt(mid) < tau - 1e-6) lo = mid + 1;
            else hi = mid;
        }
        return vt(lo);
    }
};

void result_cb(const shav_hop_result* r, void* u) {
    Part* p = static_cast<Part*>(u);
    std::lock_guard<std::mutex> lk(p->mu);
    p->res.push_back(*r);
}

struct LateHist {
    // feeder lateness of video pushes, 1 ms bins up to 4 s
    std::vector<uint64_t> bins = std::vector<uint64_t>(4001, 0);
    double max = 0;
    void add(double s) {
        const double ms = std::max(0.0, s * 1e3);
        bins[std::min<size_t>(4000, size_t(ms))]++;
        max = std::max(max, ms);
    }
    double q(double f) const {
        uint64_t n = 0;
        for (auto b : bins) n += b;
        if (!n) return NAN;
        const uint64_t want = uint64_t(std::ceil(f * double(n)));
        uint64_t c = 0;
        for (size_t i = 0; i < bins.size(); ++i)
            if ((c += bins[i]) >= want) return double(i);
        return 4000;
    }
};

void sleep_until(double t) {
    const double d = t - now_s();
    if (d > 0) {
        timespec ts{time_t(d), long((d - std::floor(d)) * 1e9)};
        nanosleep(&ts, nullptr);
    }
}

// sample-id suffix _SSSSS_EEEEE -> source duration in seconds (to prefer long clips)
double id_duration(const std::string& id) {
    const size_t b = id.rfind('_');
    if (b == std::string::npos || b < 6) return 0;
    const size_t a = id.rfind('_', b - 1);
    if (a == std::string::npos) return 0;
    return std::atof(id.c_str() + b + 1) - std::atof(id.c_str() + a + 1);
}

struct GpuMon {
    pid_t pid = -1;
    std::string path;
    void start(const std::string& p) {
        path = p;
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_addopen(&fa, 1, path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
        const char* av[] = {"nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                            "-i", "0", "-lms", "500", nullptr};
        if (posix_spawnp(&pid, av[0], &fa, nullptr, const_cast<char* const*>(av), environ) != 0) pid = -1;
        posix_spawn_file_actions_destroy(&fa);
    }
    void stop(double& util_mean, double& util_max, double& mem_max, int& samples) {
        util_mean = util_max = mem_max = NAN;
        samples = 0;
        if (pid <= 0) return;
        kill(pid, SIGTERM);   // our own child
        int st;
        waitpid(pid, &st, 0);
        std::ifstream in(path);
        std::string line;
        double su = 0;
        util_max = mem_max = 0;
        while (std::getline(in, line)) {
            double u = 0, m = 0;
            if (std::sscanf(line.c_str(), "%lf, %lf", &u, &m) != 2) continue;
            su += u;
            util_max = std::max(util_max, u);
            mem_max = std::max(mem_max, m);
            ++samples;
        }
        util_mean = samples ? su / samples : NAN;
    }
};

std::string pct_json(const std::vector<double>& v) {
    const Pct p = pct(v);
    return "{\"n\":" + std::to_string(p.n) + ",\"p50\":" + fmt(p.p50, "%.2f") + ",\"p95\":" + fmt(p.p95, "%.2f") + ",\"p99\":" +
           fmt(p.p99, "%.2f") + ",\"max\":" + fmt(p.max, "%.2f") + ",\"mean\":" + fmt(p.mean, "%.2f") + "}";
}

}  // namespace

int main(int argc, char** argv) {
    int N = 8, F = 8, FA = 0, A = 8, W = 6, K = 8, blocking = 0, max_delay = 0, video_cache = 1;
    double T = 72.0, clip_seconds = 20.0;
    std::string arrival = "burst", backend = "cuda", video_mode, out;
    bool no_video = false, no_audio = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--participants") N = std::atoi(next().c_str());
        else if (a == "--arrival") arrival = next();
        else if (a == "--seconds") T = std::atof(next().c_str());
        else if (a == "--backend") backend = next();
        else if (a == "--video-mode") video_mode = next();
        else if (a == "--feeders") F = std::atoi(next().c_str());
        else if (a == "--audio-feeders") FA = std::atoi(next().c_str());
        else if (a == "--audio-workers") A = std::atoi(next().c_str());
        else if (a == "--cpu-video-workers") W = std::atoi(next().c_str());
        else if (a == "--blocking-sync") blocking = std::atoi(next().c_str());
        else if (a == "--max-delay-us") max_delay = std::atoi(next().c_str());
        else if (a == "--clips") K = std::atoi(next().c_str());
        else if (a == "--clip-seconds") clip_seconds = std::atof(next().c_str());
        else if (a == "--video-cache") video_cache = std::atoi(next().c_str());
        else if (a == "--no-video") no_video = true;
        else if (a == "--no-audio") no_audio = true;
        else if (a == "--out") out = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (arrival != "burst" && arrival != "staggered") { std::fprintf(stderr, "--arrival burst|staggered\n"); return 2; }
    if (out.empty())
        out = "bench/artifacts/service/load/" + backend + "_" + arrival + "_n" + std::to_string(N);
    F = std::max(1, std::min(F, N));
    const std::string up0 = read_file("/proc/loadavg");
    // ---------------------------------------------------------------- engine
    shav_engine_config c;
    shav_engine_default_config(&c);
    c.audio_weights = no_audio ? "" : "build/audio/shield.weights";
    c.audio_workers = A;
    c.video_weights = no_video ? "" : "build/shield_v.weights";
    c.video_backend = backend.c_str();
    c.video_mode = video_mode.empty() ? nullptr : video_mode.c_str();
    c.max_sessions = std::max(N, 1);
    c.cpu_video_workers = W;
    c.cuda_blocking_sync = blocking;
    c.cuda_max_delay_us = max_delay;
    const double t_load0 = now_s();
    shav_engine* e = shav_engine_create(&c);
    if (!e) { std::fprintf(stderr, "engine: %s\n", shav_last_error()); return 1; }
    const double t_load = now_s() - t_load0;
    shav_engine_info info;
    shav_engine_get_info(e, &info);
    std::fprintf(stderr, "engine: video %s %s, audio %s, load %.2f s, log: %s\n", info.video_backend, info.video_mode,
                 info.audio_mode, t_load, info.load_log);
    // ---------------------------------------------------------------- inputs
    Npy16 audio;
    if (!audio.open(shvbench_data("/win_v2/test.npy"))) { std::fprintf(stderr, "audio npy\n"); return 1; }
    std::vector<Clip> clips;
    std::string clip_desc = "[";
    if (!no_video) {
        auto rows = load_map("test");
        std::stable_sort(rows.begin(), rows.end(), [](const MapRow& a, const MapRow& b) {
            return id_duration(a.sample_id) > id_duration(b.sample_id);
        });
        std::vector<const MapRow*> pick;
        std::vector<std::string> groups;
        for (int want_label : {0, 1})   // alternate Real / Fake, distinct sources
            for (const auto& r : rows) {
                if (int(pick.size()) >= K) break;
                if (r.label != want_label || std::find(groups.begin(), groups.end(), r.source_group) != groups.end()) continue;
                if (int(std::count_if(pick.begin(), pick.end(), [&](const MapRow* q) { return q->label == want_label; })) >= (K + 1) / 2)
                    break;
                pick.push_back(&r);
                groups.push_back(r.source_group);
            }
        clips.resize(pick.size());
        std::vector<std::thread> th;
        std::atomic<int> bad{0};
        for (size_t i = 0; i < pick.size(); ++i)
            th.emplace_back([&, i] {
                std::string err;
                if (!decode_clip(pick[i]->path, true, clip_seconds, clips[i], err)) bad++;
                clips[i].sample_id = pick[i]->sample_id;
            });
        for (auto& t : th) t.join();
        if (bad) { std::fprintf(stderr, "clip decode failed\n"); return 1; }
        for (size_t i = 0; i < clips.size(); ++i)
            clip_desc += std::string(i ? "," : "") + "{\"id\":\"" + clips[i].sample_id + "\",\"label\":" +
                         std::to_string(pick[i]->label) + ",\"frames\":" + std::to_string(clips[i].t.size()) + ",\"size\":\"" +
                         std::to_string(clips[i].w) + "x" + std::to_string(clips[i].h) + "\"}";
    }
    clip_desc += "]";
    size_t clip_bytes = 0;
    for (auto& cl : clips) clip_bytes += cl.data.size();
    // ---------------------------------------------------------------- sessions
    const size_t rss_before = rss_bytes();
    std::vector<std::unique_ptr<Part>> parts;
    for (int p = 0; p < N; ++p) {
        auto P = std::make_unique<Part>();
        P->id = p;
        P->pid = "load-" + std::to_string(p);
        P->row0 = (size_t(p) * 37) % audio.rows;
        if (!no_video) {
            P->clip = &clips[size_t(p) % clips.size()];
            const double dt = P->clip->t[1] - P->clip->t[0];
            P->period = P->clip->t.back() + dt;
            P->voff = std::fmod(0.0071 * p, dt);
        }
        shav_session_config sc;
        shav_session_default_config(&sc);
        sc.participant_id = P->pid.c_str();
        sc.audio_enabled = !no_audio;
        sc.video_enabled = !no_video;
        sc.video_cache = video_cache;
        sc.callback = result_cb;
        sc.user = P.get();
        P->res.reserve(size_t(T / 2) + 8);
        P->s = shav_session_create(e, &sc);
        if (!P->s) { std::fprintf(stderr, "session %d: %s\n", p, shav_last_error()); return 1; }
        parts.push_back(std::move(P));
    }
    const double stagger = arrival == "staggered" ? 2.0 : 0.0;
    const double wall0 = now_s() + 1.0;
    for (auto& P : parts) P->s0 = wall0 + stagger * double(P->id) / double(N);
    // ---------------------------------------------------------------- feeders
    // F feeders push video (and audio too unless FA > 0); FA > 0 audio feeders push only audio, the way a real server
    // receives the two modalities on different threads (push_audio and push_video of one session may run concurrently)
    if (no_audio) FA = 0;
    FA = std::max(0, std::min(FA, N));
    std::vector<LateHist> late{size_t(F)};
    std::atomic<int> feed_err{0};
    std::vector<std::thread> feeders;
    for (int f = 0; f < F + FA; ++f)
        feeders.emplace_back([&, f] {
            const bool audio_only = f >= F;
            const bool do_audio = !no_audio && (audio_only || FA == 0);
            const bool do_video = !no_video && !audio_only;
            const int k = audio_only ? f - F : f, nk = audio_only ? FA : F;
            pthread_setname_np(pthread_self(), audio_only ? "shav-afeed" : "shav-feed");
            std::vector<Part*> mine;
            for (int p = k; p < N; p += nk) mine.push_back(parts[size_t(p)].get());
            std::vector<float> buf;
            const double tick = 0.010;
            double next_tick = wall0;
            for (;;) {
                sleep_until(next_tick);
                const double now = now_s();
                bool all = true;
                for (Part* P : mine) {
                    bool& done = audio_only ? P->done_a : P->done;
                    if (done) continue;
                    all = false;
                    const double st = now - P->s0;
                    if (st < 0) continue;
                    const double lim = std::min(st, T);
                    if (do_audio) {
                        const size_t want = size_t(lim * 16000.0);
                        if (want > P->apos) {
                            buf.resize(want - P->apos);
                            for (size_t i = 0; i < buf.size(); ++i) {
                                const size_t k = P->apos + i;
                                buf[i] = float(audio.row((P->row0 + k / 160000) % audio.rows)[k % 160000]) / 32768.0f;
                            }
                            if (shav_session_push_audio(P->s, buf.data(), buf.size()) != 0) feed_err++;
                            P->apos = want;
                        }
                    }
                    if (do_video) {
                        for (;;) {
                            const double t = P->vt(P->vnext);
                            if (t > lim) break;
                            const size_t nf = P->clip->t.size();
                            late[size_t(f)].add(now_s() - (P->s0 + t));
                            if (push_frame(P->s, t, P->clip->frame(P->vnext % nf), P->clip->w, P->clip->h, true) != 0) feed_err++;
                            ++P->vnext;
                        }
                    }
                    if (st >= T) done = true;
                }
                if (all) break;
                next_tick += tick;
                if (next_tick < now_s() - tick) next_tick = now_s();   // fell behind: do not burst-catch-up ticks
            }
        });
    // ---------------------------------------------------------------- measurement window
    const double m0 = wall0 + 10.0 + stagger, m1 = wall0 + T;
    sleep_until(m0);
    GpuMon gm;
    const std::string gpath = out + "_nvsmi.csv";
    if (backend == "cuda" && !no_video) gm.start(gpath);
    auto cpu0 = thread_cpu();
    const double pc0 = process_cpu_s(), w0 = now_s();
    sleep_until(m1);
    auto cpu1 = thread_cpu();
    const double pc1 = process_cpu_s(), w1 = now_s();
    const size_t rss_end = rss_bytes();
    size_t sess_bytes = 0, sess_video = 0, sess_audio = 0, sess_crops = 0, sess_tok = 0;
    for (auto& P : parts) {
        shav_session_stats st;
        shav_session_get_stats(P->s, &st);
        sess_bytes += st.bytes_total;
        sess_video += st.bytes_video_frames;
        sess_audio += st.bytes_audio;
        sess_crops += st.bytes_crops;
        sess_tok += st.bytes_tokens_device + st.bytes_tokens_host;
    }
    shav_engine_stats es;
    shav_engine_get_stats(e, &es);
    double gu_mean, gu_max, gmem_max;
    int gsamples;
    gm.stop(gu_mean, gu_max, gmem_max, gsamples);
    const std::string up1 = read_file("/proc/loadavg");
    for (auto& t : feeders) t.join();
    {
        std::vector<std::thread> fin;
        for (int f = 0; f < F; ++f)
            fin.emplace_back([&, f] {
                for (int p = f; p < N; p += F) shav_session_finish(parts[size_t(p)]->s, 120.0);
            });
        for (auto& t : fin) t.join();
    }
    const size_t hwm = hwm_bytes();
    // ---------------------------------------------------------------- analysis
    const int64_t h_first = 4, h_last = int64_t(std::floor(T / 2.0 + 1e-9)) - 1;
    std::vector<double> la, lv, lvs, lh, vq, vc, aq, ac, bw;
    long expected = 0, emitted = 0, late_hops = 0, missing = 0, a_valid = 0, a_insuff = 0, a_other = 0, v_valid = 0,
         v_gap = 0, v_lag = 0, v_over = 0, v_fail = 0, v_other = 0, enc = 0, reuse = 0, a_miss = 0, v_miss = 0;
    for (auto& P : parts) {
        std::lock_guard<std::mutex> lk(P->mu);
        std::map<int64_t, const shav_hop_result*> by;
        for (const auto& r : P->res) by[r.hop] = &r;
        for (int64_t h = h_first; h <= h_last; ++h) {
            ++expected;
            auto it = by.find(h);
            if (it == by.end()) { ++missing; continue; }
            ++emitted;
            const shav_hop_result& r = *it->second;
            const double hop_due = P->s0 + 2.0 * double(h + 1);
            const double lat_h = r.t_emit - hop_due;
            lh.push_back(lat_h * 1e3);
            bool miss = lat_h > 2.0;
            if (!no_audio) {
                const auto& a = r.audio;
                if (a.status == SHAV_VALID) {
                    ++a_valid;
                    la.push_back((a.t_done - hop_due) * 1e3);
                    aq.push_back(a.queue_ms);
                    ac.push_back(a.compute_ms);
                    if (a.t_done - hop_due > 2.0) ++a_miss;
                } else if (a.status == SHAV_INSUFFICIENT_SPEECH) ++a_insuff;
                else { ++a_other; miss = true; }
            }
            if (!no_video) {
                const auto& v = r.video;
                if (v.status == SHAV_VALID) {
                    ++v_valid;
                    const double due = P->s0 + P->frame_at(2.0 * double(h) + 1.5);
                    lv.push_back((v.t_done - due) * 1e3);
                    lvs.push_back((v.t_done - v.t_input) * 1e3);
                    vq.push_back(v.queue_ms);
                    vc.push_back(v.compute_ms);
                    bw.push_back(v.batch_windows);
                    enc += v.frames_encoded;
                    reuse += v.frames_reused;
                    if (v.t_done - due > 2.0) ++v_miss;
                } else {
                    miss = true;
                    if (v.reason == SHAV_R_GAP) ++v_gap;
                    else if (v.reason == SHAV_R_LAG) ++v_lag;
                    else if (v.reason == SHAV_R_OVERLOAD) ++v_over;
                    else if (v.reason == SHAV_R_FAILED) ++v_fail;
                    else ++v_other;
                }
            }
            late_hops += miss;
        }
    }
    late_hops += missing;
    LateHist lt;
    for (auto& l : late) {
        for (size_t i = 0; i < lt.bins.size(); ++i) lt.bins[i] += l.bins[i];
        lt.max = std::max(lt.max, l.max);
    }
    const double wall = w1 - w0;
    std::string thr = "{";
    {
        bool first = true;
        for (auto& kv : cpu1) {
            const double d = kv.second - (cpu0.count(kv.first) ? cpu0[kv.first] : 0.0);
            if (d < 0.005 * wall) continue;
            thr += std::string(first ? "" : ",") + "\"" + kv.first + "\":" + fmt(d / wall, "%.3f");
            first = false;
        }
    }
    thr += "}";
    const double cores = (pc1 - pc0) / wall;
    std::string js = "{\"backend\":\"" + std::string(info.video_backend) + "\",\"video_mode\":\"" + info.video_mode +
                     "\",\"audio_mode\":\"" + info.audio_mode + "\",\"participants\":" + std::to_string(N) + ",\"arrival\":\"" +
                     arrival + "\",\"audio_feeders\":" + std::to_string(FA) + ",\"seconds\":" + fmt(T, "%.1f") + ",\"measured_hops\":[" + std::to_string(h_first) + "," +
                     std::to_string(h_last) + "],\"measure_window_s\":" + fmt(wall, "%.2f") + ",\"feeders\":" + std::to_string(F) +
                     ",\"audio_workers\":" + std::to_string(A) + ",\"cpu_video_workers\":" + std::to_string(W) +
                     ",\"blocking_sync\":" + std::to_string(blocking) + ",\"max_delay_us\":" + std::to_string(max_delay) +
                     ",\"video_cache\":" + std::to_string(video_cache) + ",\"no_video\":" + (no_video ? "true" : "false") +
                     ",\"no_audio\":" + (no_audio ? "true" : "false") + ",\"load_log\":\"" + info.load_log + "\"" +
                     ",\"hops\":{\"expected\":" + std::to_string(expected) + ",\"emitted\":" + std::to_string(emitted) +
                     ",\"missing\":" + std::to_string(missing) + ",\"deadline_misses\":" + std::to_string(late_hops) + "}" +
                     ",\"audio\":{\"valid\":" + std::to_string(a_valid) + ",\"insufficient_speech\":" + std::to_string(a_insuff) +
                     ",\"other\":" + std::to_string(a_other) + ",\"over_2s\":" + std::to_string(a_miss) + ",\"latency_ms\":" + pct_json(la) +
                     ",\"queue_ms\":" + pct_json(aq) + ",\"compute_ms\":" + pct_json(ac) + "}" + ",\"video\":{\"valid\":" +
                     std::to_string(v_valid) + ",\"gap\":" + std::to_string(v_gap) + ",\"lag\":" + std::to_string(v_lag) +
                     ",\"overload\":" + std::to_string(v_over) + ",\"failed\":" + std::to_string(v_fail) + ",\"other\":" +
                     std::to_string(v_other) + ",\"over_2s\":" + std::to_string(v_miss) + ",\"latency_ms\":" + pct_json(lv) +
                     ",\"service_latency_ms\":" + pct_json(lvs) + ",\"queue_ms\":" + pct_json(vq) + ",\"compute_ms\":" + pct_json(vc) +
                     ",\"batch_windows\":" + pct_json(bw) + ",\"frames_encoded\":" + std::to_string(enc) + ",\"frames_reused\":" +
                     std::to_string(reuse) + ",\"token_cache_hit_rate\":" +
                     fmt(enc + reuse ? double(reuse) / double(enc + reuse) : NAN, "%.4f") + ",\"feeder_lateness_ms\":{\"p50\":" +
                     fmt(lt.q(0.5), "%.0f") + ",\"p99\":" + fmt(lt.q(0.99), "%.0f") + ",\"max\":" + fmt(lt.max, "%.1f") + "}}" +
                     ",\"hop_latency_ms\":" + pct_json(lh) + ",\"cpu\":{\"cores\":" + fmt(cores, "%.3f") + ",\"by_thread\":" + thr +
                     "},\"gpu\":{\"util_mean\":" + fmt(gu_mean, "%.1f") + ",\"util_max\":" + fmt(gu_max, "%.0f") +
                     ",\"mem_used_max_mib\":" + fmt(gmem_max, "%.0f") + ",\"samples\":" + std::to_string(gsamples) +
                     ",\"model_device_bytes\":" + std::to_string(es.video_device_bytes) + "},\"memory\":{\"rss_before_sessions\":" +
                     std::to_string(rss_before) + ",\"rss_end\":" + std::to_string(rss_end) + ",\"rss_peak\":" + std::to_string(hwm) +
                     ",\"clip_library_bytes\":" + std::to_string(clip_bytes) + ",\"rss_per_participant\":" +
                     fmt(double(rss_end) - double(rss_before) > 0 ? (double(rss_end) - double(rss_before)) / N : NAN, "%.0f") +
                     ",\"session_bytes_per_participant\":" + fmt(double(sess_bytes) / N, "%.0f") + ",\"video_frames_per_participant\":" +
                     fmt(double(sess_video) / N, "%.0f") + ",\"audio_per_participant\":" + fmt(double(sess_audio) / N, "%.0f") +
                     ",\"crops_per_participant\":" + fmt(double(sess_crops) / N, "%.0f") + ",\"tokens_per_participant\":" +
                     fmt(double(sess_tok) / N, "%.0f") + ",\"crop_buffers\":" + std::to_string(es.crop_buffers) +
                     ",\"crop_buffer_bytes\":" + std::to_string(es.crop_buffer_bytes) + "},\"feed_errors\":" +
                     std::to_string(feed_err.load()) + ",\"engine_load_s\":" + fmt(t_load, "%.2f") + ",\"loadavg_start\":\"" +
                     up0.substr(0, up0.size() ? up0.size() - 1 : 0) + "\",\"loadavg_end\":\"" +
                     up1.substr(0, up1.size() ? up1.size() - 1 : 0) + "\",\"clips\":" + clip_desc + "}";
    std::printf("%s\n", js.c_str());
    // per-hop results of participants 0..31 (their inputs do not depend on N or the arrival pattern): cross-run
    // independence check (bench/service_load_compare.py)
    if (FILE* fh = std::fopen((out + "_hops.jsonl").c_str(), "w")) {
        for (auto& P : parts) {
            if (P->id >= 32) break;
            std::lock_guard<std::mutex> lk(P->mu);
            for (const auto& r : P->res)
                std::fprintf(fh, "{\"p\":%d,\"hop\":%lld,\"a\":\"%s/%s\",\"al\":%s,\"v\":\"%s/%s\",\"vl\":%s,\"anchors\":%d,\"alert\":%d}\n",
                             P->id, (long long)r.hop, shav_status_name(r.audio.status), shav_reason_name(r.audio.reason),
                             fmt(r.audio.logit, "%.9g").c_str(), shav_status_name(r.video.status), shav_reason_name(r.video.reason),
                             fmt(r.video.logit, "%.9g").c_str(), r.video.face_detection_anchors, r.alert);
        }
        std::fclose(fh);
    }
    FILE* fo = std::fopen((out + ".json").c_str(), "w");
    if (fo) {
        std::fprintf(fo, "%s\n", js.c_str());
        std::fclose(fo);
    }
    for (auto& P : parts) shav_session_free(P->s);
    shav_engine_free(e);
    return 0;
}
