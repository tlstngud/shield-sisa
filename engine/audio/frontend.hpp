#pragma once
#include "core.hpp"
#include <vector>
namespace shield {
// A stream caches final CNN tokens and the Mel spectrogram. General callers
// also cache PCM to verify overlap; the service guarantees its own ring shifts.
// Large CNN buffers are reused in worker-local scratch; no activations persist per stream.
struct FrontendCache {
    std::vector<float> pcm, mel, tokens;
    const Weights* weights=nullptr;
    std::uint64_t execution_mode=0;
    bool valid=false;
    // Only the service sets this: its ring guarantees a 32000-sample shift.
    // It invalidates `valid` at every VAD-skipped or failed inference hop.
    // General frontend callers retain the explicit PCM overlap comparison.
    bool trusted_shift=false;
    std::size_t reused_tokens=0, computed_mel_frames=0;
    void reset(){pcm.clear();mel.clear();tokens.clear();weights=nullptr;valid=false;}
    std::size_t bytes()const{return sizeof(*this)+(pcm.capacity()+mel.capacity()+tokens.capacity())*sizeof(float);}
};
void batchnorm_activate(float*,int rows,int channels,const float* scale,const float* shift,bool activation);
std::size_t packed_weight_bytes();
const char* frontend_kernel_name();
std::size_t frontend_scratch_bytes();
void validate_frontend_weights(const Weights&);
void release_frontend_weights(const Weights&);
void release_kernel_weights(const Weights&);
}
