// demo_av: one SHIELD-AV participant from a local video file plus an audio source; prints one JSON line per 2 s hop.
//
//   build/service/demo_av --video FILE.mp4 [--audio FILE | --audio win_v2:ROW | --audio silence]
//                         [--backend auto|cuda|cpu] [--video-mode M] [--policy either|both|audio|video]
//                         [--audio-threshold X] [--alert K/N] [--rgb] [--realtime] [--seconds T] [--id NAME]
//
// Video: decoded by ffmpeg in presentation order (native yuv420p, BT.601 limited range -> the library's exact
// I420->RGB; --rgb uses the shim's rgb24 conversion instead); timestamps = packet PTS relative to the first frame, so
// the first frame is at session time 0.  Audio: FILE is decoded by ffmpeg to 16 kHz mono s16 (first audio stream);
// win_v2:ROW plays the win_v2 test windows ROW, ROW+1, ... (int16 / 32768); silence = zeros.  Both modalities are fed in
// lockstep on the session clock (20 ms steps); --realtime paces the feeding at wall-clock rate.  Input ends with the
// shorter of the two (or --seconds).  stdout: one JSON object per hop (shav_result_json); stderr: engine info + summary.
// Run with CUDA_VISIBLE_DEVICES set (the engine falls back to the CPU backend when CUDA does not load).
#include "data_path.hpp"
#include "service_common.hpp"

using namespace svc;

namespace {
struct AudioPipe {
    pid_t pid = -1;
    int fd = -1;
    bool open(const std::string& path) {
        const char* av[] = {"ffmpeg", "-v", "error", "-nostdin", "-i", path.c_str(), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                            "-f", "s16le", "pipe:1", nullptr};
        int pfd[2];
        // O_CLOEXEC: concurrently spawned decoders must not inherit each other's pipe ends (a stray read end keeps a
        // writer blocked after the reader closed early)
        if (pipe2(pfd, O_CLOEXEC)) return false;
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, pfd[1], 1);
        posix_spawn_file_actions_addclose(&fa, pfd[0]);
        posix_spawn_file_actions_addclose(&fa, pfd[1]);
        const int rc = posix_spawnp(&pid, av[0], &fa, nullptr, const_cast<char* const*>(av), environ);
        posix_spawn_file_actions_destroy(&fa);
        close(pfd[1]);
        if (rc) { close(pfd[0]); return false; }
        fd = pfd[0];
        return true;
    }
    size_t read(int16_t* out, size_t n) {   // samples read (< n at end)
        size_t got = 0;
        uint8_t* b = reinterpret_cast<uint8_t*>(out);
        while (got < n * 2) {
            const ssize_t r = ::read(fd, b + got, n * 2 - got);
            if (r <= 0) break;
            got += size_t(r);
        }
        return got / 2;
    }
    void close_wait() {
        if (fd >= 0) ::close(fd);
        fd = -1;
        int st;
        if (pid > 0) waitpid(pid, &st, 0);
        pid = -1;
    }
};
}  // namespace

int main(int argc, char** argv) {
    std::string video, audio_src = "win_v2:0", backend = "auto", vmode, policy = "either", id = "demo";
    double athr = 0.5, max_s = 1e30;
    int ak = 1, an = 1;
    bool rgb = false, realtime = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--video") video = next();
        else if (a == "--audio") audio_src = next();
        else if (a == "--backend") backend = next();
        else if (a == "--video-mode") vmode = next();
        else if (a == "--policy") policy = next();
        else if (a == "--audio-threshold") athr = std::atof(next().c_str());
        else if (a == "--alert") { const std::string v = next(); std::sscanf(v.c_str(), "%d/%d", &ak, &an); }
        else if (a == "--rgb") rgb = true;
        else if (a == "--realtime") realtime = true;
        else if (a == "--seconds") max_s = std::atof(next().c_str());
        else if (a == "--id") id = next();
        else { std::fprintf(stderr, "unknown argument %s (see the header of tools/demo_av.cpp)\n", a.c_str()); return 2; }
    }
    if (video.empty()) { std::fprintf(stderr, "usage: demo_av --video FILE [--audio FILE|win_v2:ROW|silence] ...\n"); return 2; }
    const int pol = policy == "both" ? SHAV_POLICY_BOTH : policy == "audio" ? SHAV_POLICY_AUDIO : policy == "video" ? SHAV_POLICY_VIDEO
                                                                                                                     : SHAV_POLICY_EITHER;
    // engine
    shav_engine_config c;
    shav_engine_default_config(&c);
    c.audio_weights = "build/audio/shield.weights";
    c.audio_workers = 2;
    c.video_weights = "build/shield_v.weights";
    c.video_backend = backend.c_str();
    c.video_mode = vmode.empty() ? nullptr : vmode.c_str();
    c.max_sessions = 1;
    c.cuda_max_frames = 64;
    c.cuda_max_windows = 8;
    c.cpu_video_workers = 1;
    shav_engine* e = shav_engine_create(&c);
    if (!e) { std::fprintf(stderr, "engine: %s\n", shav_last_error()); return 1; }
    shav_engine_info info;
    shav_engine_get_info(e, &info);
    std::fprintf(stderr, "{\"engine\":\"%s\",\"video_backend\":\"%s\",\"video_mode\":\"%s\",\"video_threshold\":%.17g,"
                         "\"video_fingerprint\":\"%s\",\"audio_mode\":\"%s\",\"audio_fingerprint\":\"%s\",\"load_log\":\"%s\"}\n",
                 shav_version(), info.video_backend, info.video_mode, info.video_threshold, info.video_fingerprint, info.audio_mode,
                 info.audio_fingerprint, info.load_log);
    shav_session_config sc;
    shav_session_default_config(&sc);
    sc.participant_id = id.c_str();
    sc.policy = pol;
    sc.audio_threshold = athr;
    sc.audio_alert_k = sc.video_alert_k = ak;
    sc.audio_alert_n = sc.video_alert_n = an;
    shav_session* s = shav_session_create(e, &sc);
    if (!s) { std::fprintf(stderr, "session: %s\n", shav_last_error()); return 1; }
    // inputs
    Probe p;
    std::string err;
    if (!probe(video, p, err)) { std::fprintf(stderr, "probe %s: %s\n", video.c_str(), err.c_str()); return 1; }
    FrameReader fr;
    if (!fr.open(video, !rgb, p.width, p.height)) { std::fprintf(stderr, "ffmpeg video\n"); return 1; }
    std::vector<uint8_t> frame(fr.frame_bytes);
    const double origin = shv_pts_seconds(p.pts[0], p.tb_num, p.tb_den);
    AudioPipe ap;
    Npy16 npy;
    size_t row = 0, apos = 0;
    int mode = 0;   // 0 file, 1 win_v2, 2 silence
    if (audio_src == "silence") mode = 2;
    else if (audio_src.rfind("win_v2:", 0) == 0) {
        mode = 1;
        row = size_t(std::atoll(audio_src.c_str() + 7));
        if (!npy.open(shvbench_data("/win_v2/test.npy"))) { std::fprintf(stderr, "win_v2 test.npy\n"); return 1; }
    } else if (!ap.open(audio_src)) { std::fprintf(stderr, "ffmpeg audio\n"); return 1; }
    // lockstep feed
    std::vector<int16_t> chunk(320);
    size_t vf = 0;
    bool v_end = false, a_end = false;
    const double wall0 = now_s();
    char buf[4096];
    long hops = 0, alerts = 0;
    auto drain = [&] {
        shav_hop_result r[16];
        for (size_t n; (n = shav_session_poll(s, r, 16)) > 0;)
            for (size_t i = 0; i < n; ++i) {
                shav_result_json(&r[i], id.c_str(), buf, sizeof buf);
                std::printf("%s\n", buf);
                ++hops;
                alerts += r[i].alert == 1;
            }
        std::fflush(stdout);
    };
    for (size_t step = 1; !a_end && !v_end; ++step) {
        const double t = 0.02 * double(step);
        if (t > max_s + 1e-9) break;
        if (realtime) {
            const double d = wall0 + t - now_s();
            if (d > 0) { timespec ts{time_t(d), long((d - std::floor(d)) * 1e9)}; nanosleep(&ts, nullptr); }
        }
        // video frames with time < t
        while (vf < p.pts.size()) {
            const double ft = shv_pts_seconds(p.pts[vf], p.tb_num, p.tb_den) - origin;
            if (ft >= t) break;
            if (!fr.next(frame.data())) { v_end = true; break; }
            if (push_frame(s, ft, frame.data(), p.width, p.height, !rgb) != 0)
                std::fprintf(stderr, "video push at %.3f s: %s\n", ft, shav_last_error());
            ++vf;
        }
        if (vf >= p.pts.size()) v_end = true;
        // 20 ms of audio
        size_t n = chunk.size();
        if (mode == 0) n = ap.read(chunk.data(), chunk.size());
        else if (mode == 1)
            for (size_t i = 0; i < n; ++i, ++apos) chunk[i] = npy.row((row + apos / 160000) % npy.rows)[apos % 160000];
        else std::fill(chunk.begin(), chunk.end(), int16_t(0));
        if (n < chunk.size()) a_end = true;
        if (n && shav_session_push_audio_s16(s, chunk.data(), n) != 0) std::fprintf(stderr, "audio push: %s\n", shav_last_error());
        drain();
    }
    fr.close_wait();
    if (mode == 0) ap.close_wait();
    shav_session_finish(s, 120.0);
    drain();
    shav_session_stats st;
    shav_session_get_stats(s, &st);
    std::fprintf(stderr, "{\"summary\":{\"hops\":%ld,\"alerts\":%ld,\"video_frames_pushed\":%lld,\"sampled\":%lld,\"windows_scored\":%lld,"
                         "\"frames_encoded\":%lld,\"frames_reused\":%lld,\"session_bytes\":%zu,\"wall_s\":%.2f}}\n",
                 hops, alerts, (long long)st.video_frames_pushed, (long long)st.video_frames_sampled,
                 (long long)st.video_windows_scored, (long long)st.video_frames_encoded, (long long)st.video_frames_reused,
                 st.bytes_total, now_s() - wall0);
    shav_session_free(s);
    shav_engine_free(e);
    return 0;
}
