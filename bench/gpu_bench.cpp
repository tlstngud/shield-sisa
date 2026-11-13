// gpu_bench: end-to-end benchmarks and API tests of build/cuda/libshield_v_cuda through the public C API.
//
//   gpu_bench predict  --mode M [--batches 1,4,32,128,512] [--iters N] [--pageable]
//       per-call latency (host uint8 clips -> logits) and clips/s of shv_cuda_predict
//   gpu_bench encode   --mode M [--frames 4,20,80,320,640] [--iters N]
//       CNN alone (shv_cuda_encode_frames, tokens back to host), frames/s
//   gpu_bench live     --mode M [--participants 1,8,32,128,512] [--new 4|20] [--hops H] [--arrival burst|staggered]
//       live update per participant: `new` new frames (stored in device token slots) + 1 window classify, as one
//       async job per participant per hop; burst = all participants submit together, staggered = hops spread
//       evenly over the 2 s period at real-time rate. Latency per job (submit -> done), deadline misses (> 2 s).
//   gpu_bench slots    --mode M
//       bitwise check: sliding windows assembled from device token slots (4 new frames per hop) == predict on the
//       same 20 frames; also tokens via encode_frames == tokens stored in slots path (row independence of the API).
// All timings print JSON lines; frames come from reference crops (no decoding here).
#include "data_path.hpp"
#include "shv_cuda.h"
#include "gpu_npz.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }
std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ','))
        if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}
struct Pct { double p50, p95, p99, mean, max; };
Pct pct(std::vector<double> v) {
    if (v.empty()) return {0, 0, 0, 0, 0};
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5))]; };
    double s = 0;
    for (double x : v) s += x;
    return {at(0.5), at(0.95), at(0.99), s / double(v.size()), v.back()};
}
const size_t FB = SHV_CUDA_FRAME_BYTES, CB = 20 * FB, TF = SHV_CUDA_TOKEN_FLOATS;

std::vector<std::string> crop_paths(int n) {
    // test split crops in map order
    std::ifstream in(shvbench_data("/shieldv_codec/maps/orig.jsonl"));
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line) && int(out.size()) < n) {
        const auto p = line.find("\"sample_id\": \"");
        const auto q = line.find('"', p + 14);
        out.push_back(shvbench_data("/shieldv_codec/crops/orig_bt601/") +
                      shvbench::sha256_hex(line.substr(p + 14, q - p - 14)).substr(0, 24) + ".npz");
    }
    return out;
}
// n clips (20 frames each) into dst, cycling over the first `distinct` reference crops
void load_clips(uint8_t* dst, int n, int distinct) {
    const auto paths = crop_paths(distinct);
    std::vector<std::vector<uint8_t>> c(paths.size());
    std::vector<std::thread> th;
    for (size_t i = 0; i < paths.size(); ++i) th.emplace_back([&, i] { c[i] = shvbench::load_crop(paths[i]); });
    for (auto& t : th) t.join();
    for (int i = 0; i < n; ++i) std::memcpy(dst + size_t(i) * CB, c[size_t(i) % c.size()].data(), CB);
}
shv_cuda_model* load(const std::string& mode, int token_slots, int max_frames = 640) {
    shv_cuda_options o;
    shv_cuda_default_options(&o);
    o.token_slots = token_slots;
    o.max_frames = max_frames;
    shv_cuda_model* m = shv_cuda_load_ex("build/shield_v.weights", mode.c_str(), 0, &o);
    if (!m) { std::fprintf(stderr, "load failed: %s\n", shv_cuda_last_error()); std::exit(1); }
    std::printf("{\"event\": \"load\", \"mode\": \"%s\", \"revision\": \"%s\", \"fingerprint\": \"%s\", \"device_bytes\": %zu, "
                "\"weight_bytes\": %zu, \"token_slots\": %d}\n",
                shv_cuda_mode(m), shv_cuda_compute_revision(), shv_cuda_fingerprint(m), shv_cuda_device_bytes(m),
                shv_cuda_weight_bytes(m), token_slots);
    return m;
}

// ------------------------------------------------------------------------------------------------ predict
int cmd_predict(const std::string& mode, std::vector<int> batches, int iters, bool pageable) {
    shv_cuda_model* m = load(mode, 0);
    const int maxb = *std::max_element(batches.begin(), batches.end());
    uint8_t* clips = pageable ? static_cast<uint8_t*>(std::malloc(size_t(maxb) * CB))
                              : static_cast<uint8_t*>(shv_cuda_host_alloc(size_t(maxb) * CB));
    load_clips(clips, maxb, 64);
    std::vector<float> logits(static_cast<size_t>(maxb));
    for (int b : batches) {
        const int warm = 2, n = std::max(3, b >= 128 ? iters / 4 : iters);
        std::vector<double> lat;
        for (int it = 0; it < warm + n; ++it) {
            const auto t = Clock::now();
            if (shv_cuda_predict(m, clips, size_t(b), logits.data()) != 0) { std::fprintf(stderr, "%s\n", shv_cuda_last_error()); return 1; }
            if (it >= warm) lat.push_back(ms_since(t));
        }
        const Pct p = pct(lat);
        std::printf("{\"what\": \"predict\", \"mode\": \"%s\", \"input\": \"%s\", \"batch\": %d, \"iters\": %d, \"p50_ms\": %.3f, "
                    "\"p95_ms\": %.3f, \"p99_ms\": %.3f, \"mean_ms\": %.3f, \"clips_per_s\": %.1f, \"per_clip_ms\": %.4f}\n",
                    mode.c_str(), pageable ? "pageable" : "pinned", b, n, p.p50, p.p95, p.p99, p.mean, b / (p.p50 / 1000.0),
                    p.p50 / b);
        std::fflush(stdout);
    }
    shv_cuda_free(m);
    return 0;
}

// ------------------------------------------------------------------------------------------------ encode
int cmd_encode(const std::string& mode, std::vector<int> frames, int iters) {
    shv_cuda_model* m = load(mode, 0);
    const int maxf = *std::max_element(frames.begin(), frames.end());
    uint8_t* fr = static_cast<uint8_t*>(shv_cuda_host_alloc(size_t((maxf + 19) / 20) * CB));
    load_clips(fr, (maxf + 19) / 20, 32);
    float* tok = static_cast<float*>(shv_cuda_host_alloc(size_t(maxf) * TF * 4));
    for (int f : frames) {
        std::vector<double> lat;
        for (int it = 0; it < 2 + iters; ++it) {
            const auto t = Clock::now();
            if (shv_cuda_encode_frames(m, fr, size_t(f), tok) != 0) { std::fprintf(stderr, "%s\n", shv_cuda_last_error()); return 1; }
            if (it >= 2) lat.push_back(ms_since(t));
        }
        const Pct p = pct(lat);
        std::printf("{\"what\": \"encode\", \"mode\": \"%s\", \"frames\": %d, \"p50_ms\": %.3f, \"p95_ms\": %.3f, \"frames_per_s\": %.1f}\n",
                    mode.c_str(), f, p.p50, p.p95, f / (p.p50 / 1000.0));
        std::fflush(stdout);
    }
    shv_cuda_free(m);
    return 0;
}

// ------------------------------------------------------------------------------------------------ live
// Participant p owns token slots [p*24, p*24+24) as a ring; frame number k of p lives in slot p*24 + k % 24.
int cmd_live(const std::string& mode, std::vector<int> parts, int newf, int hops, const std::string& arrival) {
    const int maxp = *std::max_element(parts.begin(), parts.end());
    shv_cuda_model* m = load(mode, maxp * 24);
    const int distinct = 16;
    uint8_t* src = static_cast<uint8_t*>(shv_cuda_host_alloc(size_t(distinct) * CB));
    load_clips(src, distinct, distinct);
    for (int P : parts) {
        // warm-up: every participant gets its first 20 frames (hop 0 encodes 20 frames)
        std::vector<long long> nframes(size_t(P), 0);
        std::vector<float> logit(static_cast<size_t>(P));
        std::vector<double> lat, qms, gms;
        std::vector<int> bsz;
        int misses = 0;
        auto submit_hop = [&](int p, int nnew, shv_cuda_job** out) {
            std::vector<int32_t> store(static_cast<size_t>(nnew)), wslots(20);
            for (int i = 0; i < nnew; ++i) store[size_t(i)] = int32_t(p * 24 + int((nframes[size_t(p)] + i) % 24));
            const long long last = nframes[size_t(p)] + nnew;  // window = frames [last-20, last)
            for (int s = 0; s < 20; ++s) wslots[size_t(s)] = int32_t(p * 24 + int((last - 20 + s) % 24));
            shv_cuda_request r{};
            const long long f0 = nframes[size_t(p)] % (distinct * 20 - nnew);
            r.frames = src + size_t(f0) * FB;
            r.n_frames = size_t(nnew);
            r.store_slots = store.data();
            r.n_windows = 1;
            r.window_slots = wslots.data();
            r.logits_out = &logit[size_t(p)];
            *out = shv_cuda_submit(m, &r);
            if (!*out) { std::fprintf(stderr, "submit: %s\n", shv_cuda_last_error()); std::exit(1); }
            nframes[size_t(p)] = last;
        };
        {
            std::vector<shv_cuda_job*> jobs(static_cast<size_t>(P));
            for (int p = 0; p < P; ++p) submit_hop(p, 20, &jobs[size_t(p)]);
            for (auto* j : jobs) { shv_cuda_job_wait(j); shv_cuda_job_free(j); }
        }
        const auto t0 = Clock::now();
        for (int h = 0; h < hops; ++h) {
            std::vector<shv_cuda_job*> jobs(static_cast<size_t>(P));
            std::vector<Clock::time_point> ts(static_cast<size_t>(P));
            const auto hop0 = t0 + std::chrono::milliseconds(2000LL * h);
            for (int p = 0; p < P; ++p) {
                if (arrival == "staggered") std::this_thread::sleep_until(hop0 + std::chrono::microseconds(2000000LL * p / P));
                else if (p == 0) std::this_thread::sleep_until(hop0);
                ts[size_t(p)] = Clock::now();
                submit_hop(p, newf, &jobs[size_t(p)]);
            }
            for (int p = 0; p < P; ++p) {
                shv_cuda_job_wait(jobs[size_t(p)]);
                double q, r, g;
                int bf, bw;
                shv_cuda_job_timing(jobs[size_t(p)], &q, &r, &g, &bf, &bw);
                const double l = q + r;
                lat.push_back(l);
                qms.push_back(q);
                gms.push_back(g);
                bsz.push_back(bw);
                misses += l > 2000.0;
                shv_cuda_job_free(jobs[size_t(p)]);
            }
        }
        const double wall = ms_since(t0);
        const Pct pl = pct(lat), pq = pct(qms), pg = pct(gms);
        double mb = 0;
        for (int b : bsz) mb += b;
        std::printf("{\"what\": \"live\", \"mode\": \"%s\", \"participants\": %d, \"new_frames\": %d, \"arrival\": \"%s\", \"hops\": %d, "
                    "\"lat_p50_ms\": %.3f, \"lat_p95_ms\": %.3f, \"lat_p99_ms\": %.3f, \"lat_max_ms\": %.3f, \"queue_p50_ms\": %.3f, "
                    "\"batch_gpu_p50_ms\": %.3f, \"mean_batch_windows\": %.1f, \"deadline_misses\": %d, \"wall_ms\": %.1f, "
                    "\"device_bytes\": %zu, \"slot_bytes_per_participant\": %zu}\n",
                    mode.c_str(), P, newf, arrival.c_str(), hops, pl.p50, pl.p95, pl.p99, pl.max, pq.p50, pg.p50,
                    mb / double(bsz.size()), misses, wall, shv_cuda_device_bytes(m), size_t(24) * TF * 4);
        std::fflush(stdout);
    }
    shv_cuda_free(m);
    return 0;
}

// ------------------------------------------------------------------------------------------------ slots
int cmd_slots(const std::string& mode) {
    shv_cuda_model* m = load(mode, 64);
    const int nclip = 3, total = nclip * 20;
    uint8_t* src = static_cast<uint8_t*>(shv_cuda_host_alloc(size_t(nclip) * CB));
    load_clips(src, nclip, nclip);
    // reference: windows [4w, 4w+20) via predict (contiguous 20 frames)
    const int nwin = (total - 20) / 4 + 1;
    std::vector<float> ref(static_cast<size_t>(nwin)), got(size_t(nwin), std::nanf(""));
    if (shv_cuda_predict(m, src, 1, ref.data()) != 0) return 1;
    for (int w = 0; w < nwin; ++w)
        if (shv_cuda_predict(m, src + size_t(w) * 4 * FB, 1, &ref[size_t(w)]) != 0) return 1;
    // slots path: hop 0 stores frames 0..19, then 4 new frames per hop; slot of frame k = k % 24
    long long n = 0;
    std::vector<shv_cuda_job*> jobs;
    for (int w = 0; w < nwin; ++w) {
        const int nnew = w == 0 ? 20 : 4;
        std::vector<int32_t> store(static_cast<size_t>(nnew)), ws(20);
        for (int i = 0; i < nnew; ++i) store[size_t(i)] = int32_t((n + i) % 24);
        for (int s = 0; s < 20; ++s) ws[size_t(s)] = int32_t((n + nnew - 20 + s) % 24);
        shv_cuda_request r{};
        r.frames = src + size_t(n) * FB;
        r.n_frames = size_t(nnew);
        r.store_slots = store.data();
        r.n_windows = 1;
        r.window_slots = ws.data();
        r.logits_out = &got[size_t(w)];
        shv_cuda_job* j = shv_cuda_submit(m, &r);  // all hops queued at once: the scheduler coalesces them
        if (!j) { std::fprintf(stderr, "%s\n", shv_cuda_last_error()); return 1; }
        jobs.push_back(j);
        n += nnew;
    }
    int bmax = 0;
    for (auto* j : jobs) {
        shv_cuda_job_wait(j);
        int bw = 0;
        shv_cuda_job_timing(j, nullptr, nullptr, nullptr, nullptr, &bw);
        bmax = std::max(bmax, bw);
        shv_cuda_job_free(j);
    }
    int diff = 0;
    for (int w = 0; w < nwin; ++w) diff += std::memcmp(&ref[size_t(w)], &got[size_t(w)], 4) != 0;
    // tokens: encode_frames of all frames at once vs one by one
    std::vector<float> ta(size_t(total) * TF), tb(size_t(total) * TF);
    shv_cuda_encode_frames(m, src, size_t(total), ta.data());
    for (int f = 0; f < total; ++f) shv_cuda_encode_frames(m, src + size_t(f) * FB, 1, tb.data() + size_t(f) * TF);
    const bool tok_same = std::memcmp(ta.data(), tb.data(), ta.size() * 4) == 0;
    std::printf("{\"what\": \"slots\", \"mode\": \"%s\", \"windows\": %d, \"max_coalesced_windows\": %d, \"logit_bitwise_diff\": %d, "
                "\"tokens_batch60_vs_batch1_identical\": %s, \"example\": [%.7g, %.7g]}\n",
                mode.c_str(), nwin, bmax, diff, tok_same ? "true" : "false", double(ref[1]), double(got[1]));
    shv_cuda_free(m);
    return diff || !tok_same ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: gpu_bench predict|encode|live|slots --mode M ...\n"); return 2; }
    const std::string cmd = argv[1];
    std::string mode = "fp32", arrival = "burst";
    std::vector<int> batches = {1, 4, 32, 128, 512}, frames = {4, 20, 80, 320, 640}, parts = {1, 8, 32, 128, 512};
    int iters = 30, newf = 4, hops = 5;
    bool pageable = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) std::exit(2); return argv[++i]; };
        if (a == "--mode") mode = next();
        else if (a == "--batches") batches = parse_list(next());
        else if (a == "--frames") frames = parse_list(next());
        else if (a == "--participants") parts = parse_list(next());
        else if (a == "--iters") iters = std::atoi(next().c_str());
        else if (a == "--new") newf = std::atoi(next().c_str());
        else if (a == "--hops") hops = std::atoi(next().c_str());
        else if (a == "--arrival") arrival = next();
        else if (a == "--pageable") pageable = true;
        else { std::fprintf(stderr, "bad argument %s\n", a.c_str()); return 2; }
    }
    if (cmd == "predict") return cmd_predict(mode, batches, iters, pageable);
    if (cmd == "encode") return cmd_encode(mode, frames, iters);
    if (cmd == "live") return cmd_live(mode, parts, newf, hops, arrival);
    if (cmd == "slots") return cmd_slots(mode);
    std::fprintf(stderr, "unknown command\n");
    return 2;
}
