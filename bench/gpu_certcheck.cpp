// gpu_certcheck: certificate enforcement of the shipped library (build/cuda/libshield_v_cuda.so).
//   gpu_certcheck WEIGHTS [mode...]   -> one JSON line per mode: loaded or the refusal message
#include "shv_cuda.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: gpu_certcheck WEIGHTS [mode...]\n"); return 2; }
    std::vector<std::string> modes;
    for (int i = 2; i < argc; ++i) modes.push_back(argv[i]);
    if (modes.empty()) modes = {"fp32", "bf16", "fp16", "int8"};
    for (const auto& md : modes) {
        shv_cuda_options o;
        shv_cuda_default_options(&o);
        o.max_frames = 20;
        o.max_windows = 1;
        o.max_host_tokens = 20;
        shv_cuda_model* m = shv_cuda_load_ex(argv[1], md.c_str(), 0, &o);
        std::string err = m ? "" : shv_cuda_last_error();
        for (auto& c : err) if (c == '"') c = '\'';
        std::printf("{\"mode\": \"%s\", \"loaded\": %s, \"revision\": \"%s\", \"error\": \"%s\"}\n", md.c_str(), m ? "true" : "false",
                    shv_cuda_compute_revision(), err.c_str());
        shv_cuda_free(m);
    }
    return 0;
}
