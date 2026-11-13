// gpu_dump: run one reference crop through the internal CUDA engine (no graphs) and write every checked
// intermediate (frame 0 / window 0) as raw float32 into OUT_DIR, for bench/gpu_fixture_check.py.
//   gpu_dump WEIGHTS MODE CROP.npz OUT_DIR
#include "engine.hpp"
#include "gpu_npz.hpp"

#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: gpu_dump WEIGHTS MODE CROP.npz OUT_DIR\n");
        return 2;
    }
    try {
        shv::Precision p;
        if (!shv::parse_precision(argv[2], &p)) throw std::runtime_error("bad mode");
        const shv::HostModel hm = shv::load_host_model(argv[1]);
        shv::EngineConfig cfg;
        cfg.max_frames = 20;
        cfg.max_windows = 1;
        cfg.use_graphs = false;
        cfg.slots = 1;
        auto eng = shv::make_engine(hm, p, cfg);
        const std::vector<uint8_t> crop = shvbench::load_crop(argv[3]);
        const auto& sl = eng->slot(0);
        SHV_CUDA_CHECK(cudaMemcpy(sl.d_frames, crop.data(), crop.size(), cudaMemcpyHostToDevice));
        std::vector<const float*> ptrs(20);
        for (int i = 0; i < 20; ++i) ptrs[size_t(i)] = sl.d_tokens + size_t(i) * 16 * 384;
        SHV_CUDA_CHECK(cudaMemcpy(sl.d_tokptr, ptrs.data(), ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice));
        eng->set_dump_dir(argv[4]);
        eng->encode(0, 20, nullptr);
        eng->classify(0, 1, nullptr);
        SHV_CUDA_CHECK(cudaDeviceSynchronize());
        float logit = 0.0f;
        SHV_CUDA_CHECK(cudaMemcpy(&logit, sl.d_logits, 4, cudaMemcpyDeviceToHost));
        std::printf("{\"mode\": \"%s\", \"logit\": %.9g, \"sha256\": \"%s\", \"weight_bytes\": %zu, \"activation_bytes\": %zu}\n",
                    argv[2], logit, hm.sha256.c_str(), eng->weight_bytes(), eng->activation_bytes());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gpu_dump: %s\n", e.what());
        return 1;
    }
    return 0;
}
