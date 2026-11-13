// gpu_kbench: model-only timings of the internal CUDA engine (GPU-resident inputs, CUDA events, graphs on).
//   gpu_kbench WEIGHTS MODE [--frames 20,80,640] [--windows 1,4,32,128,512] [--clips 1,4,32] [--iters N]
//              [--no-graphs] [--crop CROP.npz]
// Prints one JSON object per measurement (p50/p95/mean ms over iters after warm-up).
#include "data_path.hpp"
#include "engine.hpp"
#include "gpu_npz.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ','))
        if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}
struct Stat { double p50, p95, mean, min; };
Stat stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double s = 0;
    for (double x : v) s += x;
    return {v[v.size() / 2], v[std::min(v.size() - 1, size_t(v.size() * 0.95))], s / double(v.size()), v.front()};
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: gpu_kbench WEIGHTS MODE [--frames L] [--windows L] [--clips L] [--iters N] [--no-graphs]\n");
        return 2;
    }
    std::vector<int> frames = {20, 80, 160, 320, 640}, windows = {1, 4, 32, 128, 512}, clips = {1, 4, 8, 16, 32};
    int iters = 30;
    bool graphs = true;
    std::string crop = shvbench_data("/shieldv_codec/crops/orig_bt601/example.npz");
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--frames" && i + 1 < argc) frames = parse_list(argv[++i]);
        else if (a == "--windows" && i + 1 < argc) windows = parse_list(argv[++i]);
        else if (a == "--clips" && i + 1 < argc) clips = parse_list(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--crop" && i + 1 < argc) crop = argv[++i];
        else if (a == "--no-graphs") graphs = false;
        else { std::fprintf(stderr, "bad argument %s\n", a.c_str()); return 2; }
    }
    try {
        shv::Precision p;
        if (!shv::parse_precision(argv[2], &p)) throw std::runtime_error("bad mode");
        const shv::HostModel hm = shv::load_host_model(argv[1]);
        shv::EngineConfig cfg;
        cfg.max_frames = 640;
        cfg.max_windows = 512;
        cfg.use_graphs = graphs;
        cfg.slots = 1;
        auto eng = shv::make_engine(hm, p, cfg);
        const auto& sl = eng->slot(0);
        const std::vector<uint8_t> c = shvbench::load_crop(crop);
        for (int f = 0; f < cfg.max_frames; f += 20)
            SHV_CUDA_CHECK(cudaMemcpy(sl.d_frames + size_t(f) * c.size() / 20, c.data(), c.size(), cudaMemcpyHostToDevice));
        // windows w read frames (w*20 + s) % max_frames
        std::vector<const float*> ptrs(size_t(cfg.max_windows) * 20);
        for (size_t e = 0; e < ptrs.size(); ++e) ptrs[e] = sl.d_tokens + (e % size_t(cfg.max_frames)) * 6144;
        SHV_CUDA_CHECK(cudaMemcpy(sl.d_tokptr, ptrs.data(), ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
        cudaStream_t s;
        SHV_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        cudaEvent_t e0, e1;
        SHV_CUDA_CHECK(cudaEventCreate(&e0));
        SHV_CUDA_CHECK(cudaEventCreate(&e1));
        auto timeit = [&](auto&& fn) {
            for (int w = 0; w < 3; ++w) fn();
            SHV_CUDA_CHECK(cudaStreamSynchronize(s));
            std::vector<double> v;
            for (int it = 0; it < iters; ++it) {
                SHV_CUDA_CHECK(cudaEventRecord(e0, s));
                fn();
                SHV_CUDA_CHECK(cudaEventRecord(e1, s));
                SHV_CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0;
                SHV_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                v.push_back(ms);
            }
            return stats(v);
        };
        std::printf("{\"mode\": \"%s\", \"graphs\": %d, \"iters\": %d, \"weight_bytes\": %zu, \"activation_bytes\": %zu}\n",
                    argv[2], graphs ? 1 : 0, iters, eng->weight_bytes(), eng->activation_bytes());
        for (int F : frames) {
            const Stat st = timeit([&] { eng->encode(0, F, s); });
            std::printf("{\"what\": \"encode\", \"frames\": %d, \"bucket\": %d, \"p50_ms\": %.4f, \"p95_ms\": %.4f, \"mean_ms\": %.4f, "
                        "\"frames_per_s\": %.1f}\n", F, eng->frame_bucket(F), st.p50, st.p95, st.mean, F / (st.p50 / 1000.0));
        }
        eng->encode(0, cfg.max_frames, s);
        for (int B : windows) {
            const Stat st = timeit([&] { eng->classify(0, B, s); });
            std::printf("{\"what\": \"classify\", \"windows\": %d, \"bucket\": %d, \"p50_ms\": %.4f, \"p95_ms\": %.4f, \"mean_ms\": %.4f, "
                        "\"windows_per_s\": %.1f}\n", B, eng->window_bucket(B), st.p50, st.p95, st.mean, B / (st.p50 / 1000.0));
        }
        for (int B : clips) {
            if (B * 20 > cfg.max_frames) continue;
            const Stat st = timeit([&] { eng->encode(0, B * 20, s); eng->classify(0, B, s); });
            std::printf("{\"what\": \"clip\", \"clips\": %d, \"p50_ms\": %.4f, \"p95_ms\": %.4f, \"mean_ms\": %.4f, \"clips_per_s\": %.1f}\n",
                        B, st.p50, st.p95, st.mean, B / (st.p50 / 1000.0));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gpu_kbench: %s\n", e.what());
        return 1;
    }
    return 0;
}
