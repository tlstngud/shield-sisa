// gpu_cli: logits of the 840 RTFS val+test reference crops through the public CUDA API (shv_cuda.h).
// Linked against the experimental library (uncertified modes allowed) because it produces the certificate inputs.
//
//   gpu_cli --weights build/shield_v.weights --mode bf16 --out OUT.jsonl
//           [--map MAP.jsonl] [--crops DIR] [--batch N] [--order map|reverse|shuffle:SEED] [--limit N]
//           [--api sync|async] [--inflight K] [--device D] [--max-frames N] [--no-graphs] [--threads T] [--pinned]
// Output: one JSON line per sample in map order: {"sample_id": ..., "logit": <float or null>}.
#include "data_path.hpp"
#include "shv_cuda.h"
#include "gpu_npz.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
std::string json_str(const std::string& line, const std::string& key) {
    const std::string k = "\"" + key + "\": \"";
    const auto p = line.find(k);
    if (p == std::string::npos) return "";
    const auto q = line.find('"', p + k.size());
    return line.substr(p + k.size(), q - p - k.size());
}
std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
    std::string weights = "build/shield_v.weights", mode = "fp32", out, order = "map", api = "sync";
    std::string map = shvbench_data("/shieldv_codec/maps/orig.jsonl");
    std::string crops = shvbench_data("/shieldv_codec/crops/orig_bt601");
    int batch = 4, limit = 0, device = 0, threads = 8, inflight = 4, max_frames = 640;
    bool graphs = true, pinned = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--weights") weights = next();
        else if (a == "--mode") mode = next();
        else if (a == "--out") out = next();
        else if (a == "--map") map = next();
        else if (a == "--crops") crops = next();
        else if (a == "--batch") batch = std::atoi(next().c_str());
        else if (a == "--order") order = next();
        else if (a == "--limit") limit = std::atoi(next().c_str());
        else if (a == "--api") api = next();
        else if (a == "--inflight") inflight = std::atoi(next().c_str());
        else if (a == "--device") device = std::atoi(next().c_str());
        else if (a == "--max-frames") max_frames = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--no-graphs") graphs = false;
        else if (a == "--pinned") pinned = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (out.empty() || batch < 1) { std::fprintf(stderr, "need --out and --batch >= 1\n"); return 2; }

    std::vector<std::string> ids;
    {
        std::ifstream in(map);
        if (!in) { std::fprintf(stderr, "cannot open map %s\n", map.c_str()); return 1; }
        std::string line;
        while (std::getline(in, line))
            if (!line.empty()) ids.push_back(json_str(line, "sample_id"));
    }
    if (limit > 0 && size_t(limit) < ids.size()) ids.resize(size_t(limit));
    const size_t n = ids.size(), CB = size_t(20) * SHV_CUDA_FRAME_BYTES;

    const double t0 = now_ms();
    uint8_t* data = pinned ? static_cast<uint8_t*>(shv_cuda_host_alloc(n * CB)) : static_cast<uint8_t*>(std::malloc(n * CB));
    if (!data) { std::fprintf(stderr, "allocation failed\n"); return 1; }
    std::atomic<size_t> next_i{0};
    std::atomic<int> failed{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            for (size_t i; (i = next_i++) < n;) {
                try {
                    const auto c = shvbench::load_crop(crops + "/" + shvbench::sha256_hex(ids[i]).substr(0, 24) + ".npz");
                    std::memcpy(data + i * CB, c.data(), CB);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "%s\n", e.what());
                    failed = 1;
                }
            }
        });
    for (auto& t : pool) t.join();
    if (failed) return 1;
    const double t_load = now_ms() - t0;

    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = i;
    if (order == "reverse") std::reverse(perm.begin(), perm.end());
    else if (order.rfind("shuffle:", 0) == 0) {
        std::mt19937_64 rng(std::strtoull(order.c_str() + 8, nullptr, 10));
        std::shuffle(perm.begin(), perm.end(), rng);
    } else if (order != "map") { std::fprintf(stderr, "bad --order\n"); return 2; }

    shv_cuda_options opt;
    shv_cuda_default_options(&opt);
    opt.use_graphs = graphs ? 1 : 0;
    opt.max_frames = max_frames;
    shv_cuda_model* m = shv_cuda_load_ex(weights.c_str(), mode.c_str(), device, &opt);
    if (!m) { std::fprintf(stderr, "load failed: %s\n", shv_cuda_last_error()); return 1; }

    std::vector<float> logits(n, std::nanf(""));
    const double t1 = now_ms();
    int rc = 0;
    // contiguous copies of each batch in the requested order (identity order uses the loaded buffer directly)
    const bool identity = order == "map";
    const size_t nb = (n + size_t(batch) - 1) / size_t(batch);
    std::vector<uint8_t*> bufs(nb, nullptr);
    std::vector<std::vector<float>> blog(nb);
    for (size_t b = 0; b < nb; ++b) {
        const size_t s = b * size_t(batch), e = std::min(n, s + size_t(batch));
        blog[b].assign(e - s, std::nanf(""));
        if (identity) { bufs[b] = data + s * CB; continue; }
        bufs[b] = pinned ? static_cast<uint8_t*>(shv_cuda_host_alloc((e - s) * CB)) : static_cast<uint8_t*>(std::malloc((e - s) * CB));
        for (size_t k = s; k < e; ++k) std::memcpy(bufs[b] + (k - s) * CB, data + perm[k] * CB, CB);
    }
    const double t2 = now_ms();
    if (api == "sync") {
        for (size_t b = 0; b < nb && !rc; ++b) {
            const size_t s = b * size_t(batch), e = std::min(n, s + size_t(batch));
            if (shv_cuda_predict(m, bufs[b], e - s, blog[b].data()) != 0) {
                std::fprintf(stderr, "predict failed: %s\n", shv_cuda_last_error());
                rc = 1;
            }
        }
    } else if (api == "async") {
        // one request per batch (batch*20 frames, batch windows), up to `inflight` outstanding: the scheduler
        // coalesces them into larger GPU batches, which must not change any logit.
        std::vector<int32_t> idx(size_t(batch) * 20);
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = int32_t(i);
        std::vector<shv_cuda_job*> jobs(nb, nullptr);
        size_t done = 0;
        for (size_t b = 0; b < nb && !rc; ++b) {
            const size_t s = b * size_t(batch), e = std::min(n, s + size_t(batch));
            shv_cuda_request r{};
            r.frames = bufs[b];
            r.n_frames = (e - s) * 20;
            r.n_windows = e - s;
            r.window_frames = idx.data();
            r.logits_out = blog[b].data();
            jobs[b] = shv_cuda_submit(m, &r);
            if (!jobs[b]) { std::fprintf(stderr, "submit failed: %s\n", shv_cuda_last_error()); rc = 1; break; }
            while (b + 1 - done >= size_t(std::max(1, inflight))) {
                if (shv_cuda_job_wait(jobs[done]) != 0) { std::fprintf(stderr, "job failed: %s\n", shv_cuda_job_error(jobs[done])); rc = 1; }
                ++done;
            }
        }
        for (size_t b = done; b < nb; ++b)
            if (jobs[b] && shv_cuda_job_wait(jobs[b]) != 0) { std::fprintf(stderr, "job failed: %s\n", shv_cuda_job_error(jobs[b])); rc = 1; }
        for (auto* j : jobs) shv_cuda_job_free(j);
    } else {
        std::fprintf(stderr, "bad --api\n");
        rc = 2;
    }
    const double t3 = now_ms();
    for (size_t b = 0; b < nb; ++b) {
        const size_t s = b * size_t(batch), e = std::min(n, s + size_t(batch));
        for (size_t k = s; k < e; ++k) logits[perm[k]] = blog[b][k - s];
    }
    std::ofstream o(out);
    for (size_t i = 0; i < n; ++i) {
        char v[64];
        if (std::isfinite(logits[i])) std::snprintf(v, sizeof v, "%.9g", double(logits[i]));
        else std::snprintf(v, sizeof v, "null");
        o << "{\"sample_id\": \"" << esc(ids[i]) << "\", \"logit\": " << v << "}\n";
    }
    std::fprintf(stderr,
                 "{\"mode\": \"%s\", \"fingerprint\": \"%s\", \"revision\": \"%s\", \"n\": %zu, \"batch\": %d, \"order\": \"%s\", "
                 "\"api\": \"%s\", \"load_crops_ms\": %.1f, \"gather_ms\": %.1f, \"infer_ms\": %.1f, \"clips_per_s\": %.1f, "
                 "\"device_bytes\": %zu}\n",
                 shv_cuda_mode(m), shv_cuda_fingerprint(m), shv_cuda_compute_revision(), n, batch, order.c_str(), api.c_str(),
                 t_load, t2 - t1, t3 - t2, double(n) / ((t3 - t2) / 1000.0), shv_cuda_device_bytes(m));
    shv_cuda_free(m);
    return rc;
}
