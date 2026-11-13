#pragma once
#include <cstddef>
#include <cstdint>

namespace shield {
struct Weights;
// Experimental modes affect linear weights/inputs only. Accumulation, bias,
// normalization, FFT, depthwise convolution and recurrent state remain FP32.
// Exact is the default in every newly created worker thread.
enum class ComputeMode : int {
    Exact=0, Fp16=1, Bf16=2, Fp16Cnn=3, Bf16Cnn=4, Int8Cnn=5
};
ComputeMode get_compute_mode() noexcept;
ComputeMode set_compute_mode(ComputeMode) noexcept;
std::uint64_t execution_mode_key() noexcept;
const char* compute_mode_name(ComputeMode) noexcept;
const char* compute_revision() noexcept;
bool compute_mode_accelerated(ComputeMode) noexcept;
void register_compute_weights(const Weights&);
struct ScopedComputeMode {
    ComputeMode prior;
    explicit ScopedComputeMode(ComputeMode mode) noexcept:prior(set_compute_mode(mode)){}
    ~ScopedComputeMode(){set_compute_mode(prior);}
    ScopedComputeMode(const ScopedComputeMode&)=delete;
    ScopedComputeMode& operator=(const ScopedComputeMode&)=delete;
};
namespace detail {
bool approximate_linear(const float*,float*,int,int,int,const float*,const float*);
void release_precision_weights(const Weights&);
std::size_t precision_weight_bytes();
}
}
