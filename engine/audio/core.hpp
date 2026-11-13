#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace shield {
struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
};
struct Weights {
    std::unordered_map<std::string, Tensor> tensors;
    std::string fingerprint;
    const Tensor& at(const std::string& name) const { return tensors.at(name); }
    std::size_t bytes() const;
    explicit Weights(const std::string& path);
};
void linear(const float* input, float* output, int rows, int in, int out,
            const float* weight, const float* bias = nullptr);
const char* kernel_name();
struct FrontendCache;
void frontend_cnn(const Weights&, const float* pcm, std::vector<float>& tokens,
                  FrontendCache* cache = nullptr);
void sequence(const Weights&, const float* tokens, float& logit);
void validate_sequence_weights(const Weights&);
}
