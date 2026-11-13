// service_audio_parity: audio results of the integrated AV service, for an independent check against the audio engine
// (bench/service_audio_parity.py: shield_predict on scipy low-passed windows + a reference shield_stream).
//
//   service_audio_parity [--backend cuda|cpu|none] [--participants P] [--seconds T]
//                        [--out bench/artifacts/service/audio_parity_hops.jsonl]
//
// P participants run concurrently (one feeding thread each) through one engine, so the audio worker pool and the GPU
// are shared.  Participant p: audio = win_v2 test rows row0(p), row0(p)+1, ... concatenated (int16 / 32768), pushed in
// pseudo-random chunk sizes (1..48000 samples; float32 API for even p, int16 API for odd p); participant 1 has zeros in
// [20 s, 34 s) (insufficient speech hops); even participants also get video (an RTFS test clip, I420, lockstep with
// the audio clock); alert k-of-n = 2 of 3, threshold 0.5.  Output: one "spec" line per participant, one line per hop.
#include "data_path.hpp"
#include "service_common.hpp"

#include <mutex>
#include <thread>

using namespace svc;

namespace {
struct Collector {
    std::mutex mu;
    std::vector<shav_hop_result> r;
};
void collect_cb(const shav_hop_result* r, void* u) {
    Collector* c = static_cast<Collector*>(u);
    std::lock_guard<std::mutex> lk(c->mu);
    c->r.push_back(*r);
}
}  // namespace

int main(int argc, char** argv) {
    std::string backend = "cuda", out = "bench/artifacts/service/audio_parity_hops.jsonl";
    int P = 8;
    double T = 60.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--backend") backend = next();
        else if (a == "--participants") P = std::atoi(next().c_str());
        else if (a == "--seconds") T = std::atof(next().c_str());
        else if (a == "--out") out = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    shav_engine_config c;
    shav_engine_default_config(&c);
    c.audio_weights = "build/audio/shield.weights";
    c.audio_workers = 4;
    c.video_weights = backend == "none" ? "" : "build/shield_v.weights";
    c.video_backend = backend.c_str();
    c.max_sessions = 64;
    shav_engine* e = shav_engine_create(&c);
    if (!e) { std::fprintf(stderr, "engine: %s\n", shav_last_error()); return 1; }
    shav_engine_info info;
    shav_engine_get_info(e, &info);
    Npy16 audio;
    if (!audio.open(shvbench_data("/win_v2/test.npy"))) { std::fprintf(stderr, "audio npy\n"); return 1; }
    Clip clip;
    if (backend != "none") {
        auto rows = load_map("test");
        std::string err;
        if (!decode_clip(rows[3].path, true, 1e9, clip, err)) { std::fprintf(stderr, "decode: %s\n", err.c_str()); return 1; }
    }
    const size_t total = size_t(std::llround(T * 16000));
    const size_t np = size_t(P);
    std::vector<Collector> cols(np);
    std::vector<std::string> ids(np);
    std::vector<size_t> row0(np);
    std::vector<std::thread> th;
    for (int p = 0; p < P; ++p) {
        ids[size_t(p)] = "audio-parity-" + std::to_string(p);
        row0[size_t(p)] = (size_t(p) * 53 + 11) % audio.rows;
    }
    for (int p = 0; p < P; ++p)
        th.emplace_back([&, p] {
            shav_session_config sc;
            shav_session_default_config(&sc);
            sc.participant_id = ids[size_t(p)].c_str();
            sc.audio_alert_k = 2;
            sc.audio_alert_n = 3;
            sc.video_enabled = backend != "none" && p % 2 == 0;
            sc.callback = collect_cb;
            sc.user = &cols[size_t(p)];
            shav_session* s = shav_session_create(e, &sc);
            if (!s) { std::fprintf(stderr, "session: %s\n", shav_last_error()); std::exit(1); }
            std::vector<float> f;
            std::vector<int16_t> q;
            uint64_t rng = 0x9E3779B97F4A7C15ull * uint64_t(p + 1);
            size_t pos = 0, vf = 0;
            const double period = clip.t.empty() ? 0 : clip.t.back() + (clip.t[1] - clip.t[0]);
            while (pos < total) {
                rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                const size_t n = std::min(total - pos, size_t(1 + (rng >> 33) % 48000));
                q.resize(n);
                for (size_t i = 0; i < n; ++i) {
                    const size_t k = pos + i;
                    int16_t v = audio.row((row0[size_t(p)] + k / 160000) % audio.rows)[k % 160000];
                    if (p == 1 && k >= 20 * 16000 && k < 34 * 16000) v = 0;
                    q[i] = v;
                }
                int rc;
                if (p % 2 == 0) {
                    f.resize(n);
                    for (size_t i = 0; i < n; ++i) f[i] = float(q[i]) / 32768.0f;
                    rc = shav_session_push_audio(s, f.data(), n);
                } else {
                    rc = shav_session_push_audio_s16(s, q.data(), n);
                }
                if (rc != 0) { std::fprintf(stderr, "push audio: %s\n", shav_last_error()); std::exit(1); }
                pos += n;
                if (sc.video_enabled) {
                    const double now = double(pos) / 16000.0;
                    for (;;) {
                        const double t = double(vf / clip.t.size()) * period + clip.t[vf % clip.t.size()];
                        if (t >= now) break;
                        if (push_frame(s, t, clip.frame(vf % clip.t.size()), clip.w, clip.h, true) != 0) {
                            std::fprintf(stderr, "push video: %s\n", shav_last_error());
                            std::exit(1);
                        }
                        ++vf;
                    }
                }
            }
            shav_session_finish(s, 300.0);
            shav_session_free(s);
        });
    for (auto& t : th) t.join();
    FILE* fo = std::fopen(out.c_str(), "w");
    if (!fo) return 1;
    std::fprintf(fo, "{\"kind\":\"engine\",\"audio_mode\":\"%s\",\"audio_fingerprint\":\"%s\",\"audio_revision\":\"%s\","
                     "\"video_backend\":\"%s\",\"video_mode\":\"%s\",\"seconds\":%.3f}\n",
                 info.audio_mode, info.audio_fingerprint, info.audio_revision, info.video_backend, info.video_mode, T);
    long hops = 0, valid = 0, vvalid = 0;
    for (int p = 0; p < P; ++p) {
        std::fprintf(fo, "{\"kind\":\"spec\",\"participant\":%d,\"id\":\"%s\",\"row0\":%zu,\"samples\":%zu,\"zero_from\":%d,"
                         "\"zero_to\":%d,\"alert_k\":2,\"alert_n\":3,\"threshold\":0.5,\"video\":%d,\"api\":\"%s\"}\n",
                     p, ids[size_t(p)].c_str(), row0[size_t(p)], total, p == 1 ? 20 * 16000 : 0, p == 1 ? 34 * 16000 : 0,
                     backend != "none" && p % 2 == 0, p % 2 == 0 ? "f32" : "s16");
        for (const auto& h : cols[size_t(p)].r) {
            const auto& a = h.audio;
            std::fprintf(fo, "{\"kind\":\"hop\",\"participant\":%d,\"hop\":%lld,\"status\":\"%s\",\"reason\":\"%s\",\"logit\":%s,"
                             "\"score\":%s,\"valid_speech_sec\":%s,\"decision\":%d,\"alert\":%d,\"window\":[%.17g,%.17g],"
                             "\"video_status\":\"%s\",\"video_reason\":\"%s\",\"combined_alert\":%d}\n",
                         p, (long long)h.hop, shav_status_name(a.status), shav_reason_name(a.reason), fmt(a.logit, "%.9g").c_str(),
                         fmt(a.score, "%.17g").c_str(), fmt(a.valid_speech_sec, "%.17g").c_str(), a.decision, a.alert,
                         a.window_start, a.window_end, shav_status_name(h.video.status), shav_reason_name(h.video.reason), h.alert);
            ++hops;
            valid += a.status == SHAV_VALID;
            vvalid += h.video.status == SHAV_VALID;
        }
    }
    std::fclose(fo);
    std::printf("{\"event\":\"done\",\"participants\":%d,\"hops\":%ld,\"audio_valid\":%ld,\"video_valid\":%ld,\"out\":\"%s\"}\n", P,
                hops, valid, vvalid, out.c_str());
    shav_engine_free(e);
    return 0;
}
