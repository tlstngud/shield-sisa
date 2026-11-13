// service_video_parity: live-API parity on RTFS videos.
// Every video is decoded frame by frame (ffmpeg, exactly the shim's BT.601 path, or native yuv420p) and pushed with its
// presentation timestamps into a video-only shav_session in reference mode (video_start = the map's window_start,
// window 0 = hop 4).  For hop 4 the crop tap must give the 20 reference crops bit for bit (crops/orig_bt601), the sampled
// times / anchor count must equal the published run, and the session's logit must equal the same engine's batch
// predict on the reference crops (bitwise).  Several backends run side by side on the same decoded frames.
//
// --all-windows: video_start = fmod(window_start, 2) instead, so the reference window is participant window
// w_ref = (window_start - video_start) / 2 (hop 4 + w_ref; the targets are the same doubles), and the whole video is
// pushed: every other scored window must also equal batch predict on its own crops (bitwise, with the exact token
// cache reusing tokens across windows), and the cache hit rate over all windows of all videos is reported.
//
//   service_video_parity [--backends cuda,cpu] [--cuda-mode fp16] [--cpu-mode exact] [--split all|val|test]
//                        [--limit N] [--workers W] [--yuv] [--all-windows] [--out PREFIX]
// Writes PREFIX.jsonl (one line per video), PREFIX_<backend>_logits.jsonl ({sample_id, logit}) and prints a summary.
#include "service_common.hpp"

#include <atomic>
#include <mutex>
#include <thread>

using namespace svc;

namespace {
struct Tap {
    int64_t ref_hop = 4;
    bool all = false;
    std::vector<uint8_t> crops;   // reference hop
    std::vector<double> times;
    int anchors = -1;
    bool got = false;
    std::map<int64_t, std::vector<uint8_t>> win;   // --all-windows: crops of every scored hop
};
void tap_cb(const shav_window_view* v, void* user) {
    Tap* t = static_cast<Tap*>(user);
    if (t->all) t->win[v->hop].assign(v->crops, v->crops + 20 * size_t(172800));
    if (v->hop != t->ref_hop || t->got) return;
    t->crops.assign(v->crops, v->crops + 20 * size_t(172800));
    t->times.assign(v->times, v->times + 20);
    t->anchors = v->anchors;
    t->got = true;
}
struct PerEngine {
    std::string name;
    shav_engine* e = nullptr;
};
struct Row {
    std::string sample_id;
    struct R {
        int status = -1, reason = -1, anchors = -1;
        float logit = NAN, batch = NAN;
        long px_diff = -1;
        bool times_ok = false, anchors_ok = false, tap = false;
        // --all-windows
        long w_valid = 0, w_eq = 0, w_unavail = 0, w_other = 0, enc = 0, reuse = 0;
    };
    std::vector<R> r;
    int frames_pushed = 0, probe_frames = 0;
    double ms = 0;
    std::string err;
};
std::vector<std::string> split_list(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ','))
        if (!x.empty()) v.push_back(x);
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::string backends = "cuda,cpu", cuda_mode = "fp16", cpu_mode = "exact", split = "all",
                out = "bench/artifacts/service/video_parity";
    int limit = 0, workers = 8;
    bool yuv = false, all_windows = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--backends") backends = next();
        else if (a == "--cuda-mode") cuda_mode = next();
        else if (a == "--cpu-mode") cpu_mode = next();
        else if (a == "--split") split = next();
        else if (a == "--limit") limit = std::atoi(next().c_str());
        else if (a == "--workers") workers = std::atoi(next().c_str());
        else if (a == "--yuv") yuv = true;
        else if (a == "--all-windows") all_windows = true;
        else if (a == "--out") out = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::vector<PerEngine> engines;
    for (const auto& b : split_list(backends)) {
        shav_engine_config c;
        shav_engine_default_config(&c);
        c.video_weights = "build/shield_v.weights";
        c.video_backend = b.c_str();
        c.video_mode = b == "cuda" ? cuda_mode.c_str() : cpu_mode.c_str();
        c.max_sessions = 64;
        c.cpu_video_workers = 4;
        c.audio_workers = 1;
        PerEngine pe;
        pe.name = b;
        pe.e = shav_engine_create(&c);
        if (!pe.e) { std::fprintf(stderr, "engine %s: %s\n", b.c_str(), shav_last_error()); return 1; }
        shav_engine_info info;
        shav_engine_get_info(pe.e, &info);
        std::printf("{\"event\":\"engine\",\"backend\":\"%s\",\"mode\":\"%s\",\"fingerprint\":\"%s\",\"revision\":\"%s\","
                    "\"threshold\":%.17g,\"load_log\":\"%s\"}\n",
                    info.video_backend, info.video_mode, info.video_fingerprint, info.video_revision, info.video_threshold,
                    info.load_log);
        engines.push_back(pe);
    }
    std::vector<MapRow> rows = load_map(split);
    if (limit > 0 && size_t(limit) < rows.size()) rows.resize(size_t(limit));
    std::vector<Row> res(rows.size());
    std::atomic<size_t> next_row{0};
    std::mutex log_mu;
    const double t_start = now_s();
    auto work = [&] {
        std::vector<uint8_t> frame;
        for (;;) {
            const size_t i = next_row.fetch_add(1);
            if (i >= rows.size()) return;
            const MapRow& m = rows[i];
            Row& R = res[i];
            R.sample_id = m.sample_id;
            R.r.resize(engines.size());
            const double t0 = now_s();
            Probe p;
            std::string err;
            if (!probe(m.path, p, err)) { R.err = err; continue; }
            R.probe_frames = int(p.pts.size());
            std::vector<Tap> taps(engines.size());
            std::vector<shav_session*> ss;
            const double vstart = all_windows ? std::fmod(m.window_start, 2.0) : m.window_start;
            const int64_t ref_hop = 4 + (all_windows ? int64_t(std::llround((m.window_start - vstart) / 2.0)) : 0);
            for (size_t k = 0; k < engines.size(); ++k) {
                taps[k].ref_hop = ref_hop;
                taps[k].all = all_windows;
                shav_session_config sc;
                shav_session_default_config(&sc);
                sc.participant_id = m.sample_id.c_str();
                sc.audio_enabled = 0;
                sc.video_start = vstart;
                sc.video_first_hop = 4;
                sc.video_max_pending = 64;
                sc.crop_tap = tap_cb;
                sc.crop_tap_user = &taps[k];
                shav_session* s = shav_session_create(engines[k].e, &sc);
                if (!s) { R.err = shav_last_error(); break; }
                ss.push_back(s);
            }
            if (ss.size() != engines.size()) { for (auto* s : ss) shav_session_free(s); continue; }
            FrameReader fr;
            if (!fr.open(m.path, yuv, p.width, p.height)) { R.err = "ffmpeg"; for (auto* s : ss) shav_session_free(s); continue; }
            frame.resize(fr.frame_bytes);
            std::vector<shav_hop_result> got(engines.size());
            std::vector<bool> have(engines.size(), false), stop(engines.size(), false);
            std::vector<std::vector<shav_hop_result>> hops(engines.size());
            size_t done = 0;   // sessions that stopped taking frames (push error, or reference hop seen without --all-windows)
            for (size_t f = 0; f < p.pts.size() && done < engines.size(); ++f) {
                if (!fr.next(frame.data())) { R.err = "decode ended at frame " + std::to_string(f); break; }
                const double t = shv_pts_seconds(p.pts[f], p.tb_num, p.tb_den);
                for (size_t k = 0; k < engines.size(); ++k) {
                    if (stop[k]) continue;
                    if (push_frame(ss[k], t, frame.data(), p.width, p.height, yuv) != 0) {
                        R.err = std::string("push: ") + shav_last_error();
                        stop[k] = true;
                        ++done;
                        continue;
                    }
                    shav_hop_result hr[8];
                    const size_t n = shav_session_poll(ss[k], hr, 8);
                    for (size_t q = 0; q < n; ++q) {
                        hops[k].push_back(hr[q]);
                        if (hr[q].hop == ref_hop) {
                            got[k] = hr[q];
                            have[k] = true;
                            if (!all_windows) { stop[k] = true; ++done; }
                        }
                    }
                }
                R.frames_pushed = int(f + 1);
            }
            fr.close_wait();
            for (size_t k = 0; k < engines.size(); ++k) {
                if (!have[k] || all_windows) {
                    shav_session_wait_idle(ss[k], 120.0);
                    shav_hop_result hr[64];
                    for (size_t n; (n = shav_session_poll(ss[k], hr, 64)) > 0;)
                        for (size_t q = 0; q < n; ++q) {
                            hops[k].push_back(hr[q]);
                            if (hr[q].hop == ref_hop) { got[k] = hr[q]; have[k] = true; }
                        }
                }
                if (all_windows) {
                    // every scored window: live logit (token cache) == batch predict on the window's own crops
                    Row::R& o = R.r[k];
                    std::vector<const shav_hop_result*> sc;
                    for (const auto& h : hops[k]) {
                        if (h.hop < 4) continue;
                        if (h.video.status == SHAV_VALID && taps[k].win.count(h.hop)) sc.push_back(&h);
                        else if (h.video.status == SHAV_UNAVAILABLE) ++o.w_unavail;
                        else ++o.w_other;
                    }
                    std::vector<uint8_t> clips(sc.size() * 20 * size_t(172800));
                    for (size_t q = 0; q < sc.size(); ++q) {
                        const auto& c = taps[k].win[sc[q]->hop];
                        std::memcpy(clips.data() + q * c.size(), c.data(), c.size());
                    }
                    std::vector<float> lg(sc.size(), NAN);
                    if (!sc.empty() && shav_engine_video_predict(engines[k].e, clips.data(), sc.size(), lg.data()) != 0)
                        R.err = std::string("predict: ") + shav_last_error();
                    for (size_t q = 0; q < sc.size(); ++q) {
                        ++o.w_valid;
                        o.w_eq += std::memcmp(&lg[q], &sc[q]->video.logit, 4) == 0;
                        o.enc += sc[q]->video.frames_encoded;
                        o.reuse += sc[q]->video.frames_reused;
                    }
                    taps[k].win.clear();
                }
                Row::R& o = R.r[k];
                if (have[k]) {
                    o.status = got[k].video.status;
                    o.reason = got[k].video.reason;
                    o.logit = got[k].video.logit;
                    o.anchors = got[k].video.face_detection_anchors;
                }
                o.tap = taps[k].got;
                shav_session_free(ss[k]);
            }
            // compare with the reference crops / published sampling
            try {
                const std::vector<uint8_t> ref = shvbench::load_crop(crop_path(m.sample_id));
                for (size_t k = 0; k < engines.size(); ++k) {
                    Row::R& o = R.r[k];
                    if (!taps[k].got) continue;
                    long d = 0;
                    for (size_t b = 0; b < ref.size(); ++b) d += ref[b] != taps[k].crops[b];
                    o.px_diff = d;
                    o.times_ok = taps[k].times.size() == m.pub_times.size();
                    for (size_t q = 0; o.times_ok && q < m.pub_times.size(); ++q) o.times_ok = taps[k].times[q] == m.pub_times[q];
                    o.anchors_ok = taps[k].anchors == m.pub_anchors && o.anchors == m.pub_anchors;
                }
            } catch (const std::exception& ex) {
                R.err = ex.what();
            }
            R.ms = (now_s() - t0) * 1e3;
            if (i % 50 == 0) {
                std::lock_guard<std::mutex> lk(log_mu);
                std::fprintf(stderr, "[%zu/%zu] %s %.0f ms\n", i, rows.size(), m.sample_id.c_str(), R.ms);
            }
        }
    };
    std::vector<std::thread> th;
    for (int w = 0; w < workers; ++w) th.emplace_back(work);
    for (auto& t : th) t.join();
    const double wall = now_s() - t_start;
    // batch predict on the reference crops, same engines
    for (size_t k = 0; k < engines.size(); ++k) {
        const size_t B = 16;
        std::vector<uint8_t> clips(B * 20 * 172800);
        for (size_t i0 = 0; i0 < rows.size(); i0 += B) {
            const size_t nb = std::min(B, rows.size() - i0);
            for (size_t q = 0; q < nb; ++q) {
                const auto c = shvbench::load_crop(crop_path(rows[i0 + q].sample_id));
                std::memcpy(clips.data() + q * c.size(), c.data(), c.size());
            }
            std::vector<float> lg(nb);
            if (shav_engine_video_predict(engines[k].e, clips.data(), nb, lg.data()) != 0) {
                std::fprintf(stderr, "predict: %s\n", shav_last_error());
                return 1;
            }
            for (size_t q = 0; q < nb; ++q) res[i0 + q].r[k].batch = lg[q];
        }
    }
    // outputs
    FILE* fo = std::fopen((out + ".jsonl").c_str(), "w");
    std::vector<FILE*> fl;
    for (auto& pe : engines) fl.push_back(std::fopen((out + "_" + pe.name + "_logits.jsonl").c_str(), "w"));
    struct Sum {
        long n = 0, crops_eq = 0, px = 0, times_ok = 0, anchors_ok = 0, valid = 0, bitwise = 0, missing = 0;
        long w_valid = 0, w_eq = 0, w_unavail = 0, w_other = 0, enc = 0, reuse = 0;
        double maxd = 0;
    };
    std::vector<Sum> sum(engines.size());
    long errors = 0;
    std::vector<std::string> bad;
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& R = res[i];
        errors += !R.err.empty();
        std::string line = "{\"sample_id\":\"" + R.sample_id + "\",\"frames_pushed\":" + std::to_string(R.frames_pushed) +
                           ",\"probe_frames\":" + std::to_string(R.probe_frames) + ",\"ms\":" + fmt(R.ms, "%.1f") +
                           ",\"err\":\"" + R.err + "\"";
        for (size_t k = 0; k < engines.size(); ++k) {
            const Row::R& o = R.r.size() > k ? R.r[k] : Row::R();
            Sum& S = sum[k];
            ++S.n;
            S.crops_eq += o.px_diff == 0;
            S.px += o.px_diff > 0 ? o.px_diff : 0;
            S.times_ok += o.times_ok;
            S.anchors_ok += o.anchors_ok;
            S.valid += o.status == SHAV_VALID;
            const bool bw = std::isfinite(o.logit) && std::memcmp(&o.logit, &o.batch, 4) == 0;
            S.bitwise += bw;
            if (std::isfinite(o.logit) && std::isfinite(o.batch)) S.maxd = std::max(S.maxd, double(std::fabs(o.logit - o.batch)));
            S.w_valid += o.w_valid;
            S.w_eq += o.w_eq;
            S.w_unavail += o.w_unavail;
            S.w_other += o.w_other;
            S.enc += o.enc;
            S.reuse += o.reuse;
            if (o.px_diff != 0 || !o.times_ok || !o.anchors_ok || !bw || o.w_eq != o.w_valid)
                bad.push_back(engines[k].name + ":" + R.sample_id);
            line += ",\"" + engines[k].name + "\":{\"status\":\"" + shav_status_name(o.status) + "\",\"reason\":\"" +
                    shav_reason_name(o.reason) + "\",\"logit\":" + fmt(o.logit, "%.9g") + ",\"batch_logit\":" +
                    fmt(o.batch, "%.9g") + ",\"bitwise\":" + (bw ? "true" : "false") + ",\"anchors\":" +
                    std::to_string(o.anchors) + ",\"px_diff\":" + std::to_string(o.px_diff) + ",\"times_ok\":" +
                    (o.times_ok ? "true" : "false") + ",\"anchors_ok\":" + (o.anchors_ok ? "true" : "false");
            if (all_windows)
                line += ",\"windows_valid\":" + std::to_string(o.w_valid) + ",\"windows_eq_batch\":" + std::to_string(o.w_eq) +
                        ",\"windows_unavailable\":" + std::to_string(o.w_unavail) + ",\"windows_other\":" + std::to_string(o.w_other) +
                        ",\"frames_encoded\":" + std::to_string(o.enc) + ",\"frames_reused\":" + std::to_string(o.reuse);
            line += "}";
            std::fprintf(fl[k], "{\"sample_id\": \"%s\", \"logit\": %s}\n", R.sample_id.c_str(),
                         o.status == SHAV_VALID ? fmt(o.logit, "%.9g").c_str() : "null");
        }
        line += "}\n";
        std::fputs(line.c_str(), fo);
    }
    std::fclose(fo);
    for (FILE* f : fl) std::fclose(f);
    std::string s = "{\"event\":\"summary\",\"videos\":" + std::to_string(rows.size()) + ",\"input\":\"" + (yuv ? "i420" : "rgb24") +
                    "\",\"mode\":\"" + (all_windows ? "all_windows" : "reference_window") +
                    "\",\"wall_s\":" + fmt(wall, "%.1f") + ",\"errors\":" + std::to_string(errors);
    for (size_t k = 0; k < engines.size(); ++k) {
        const Sum& S = sum[k];
        shav_engine_info info;
        shav_engine_get_info(engines[k].e, &info);
        s += ",\"" + engines[k].name + "\":{\"mode\":\"" + info.video_mode + "\",\"crops_bit_identical\":" + std::to_string(S.crops_eq) +
             ",\"pixels_differing\":" + std::to_string(S.px) + ",\"times_equal\":" + std::to_string(S.times_ok) +
             ",\"anchors_equal\":" + std::to_string(S.anchors_ok) + ",\"valid\":" + std::to_string(S.valid) +
             ",\"logit_bitwise_equal_batch\":" + std::to_string(S.bitwise) + ",\"max_abs_live_minus_batch\":" + fmt(S.maxd, "%.3g");
        if (all_windows)
            s += ",\"all_windows\":{\"valid\":" + std::to_string(S.w_valid) + ",\"eq_batch\":" + std::to_string(S.w_eq) +
                 ",\"unavailable\":" + std::to_string(S.w_unavail) + ",\"other\":" + std::to_string(S.w_other) +
                 ",\"frames_encoded\":" + std::to_string(S.enc) + ",\"frames_reused\":" + std::to_string(S.reuse) +
                 ",\"token_cache_hit_rate\":" + fmt(S.enc + S.reuse ? double(S.reuse) / double(S.enc + S.reuse) : NAN, "%.5f") + "}";
        s += "}";
    }
    s += ",\"mismatches\":[";
    for (size_t i = 0; i < bad.size() && i < 40; ++i) s += (i ? ",\"" : "\"") + bad[i] + "\"";
    s += "]}";
    std::printf("%s\n", s.c_str());
    for (auto& pe : engines) shav_engine_free(pe.e);
    return bad.empty() && errors == 0 ? 0 : 1;
}
