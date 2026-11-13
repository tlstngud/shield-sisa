// SHIELD-V CUDA engine core (internal C++ interface used by api.cpp and the bench tools).
#pragma once
#include "common.cuh"
#include "model.hpp"

#include <memory>
#include <string>
#include <vector>

namespace shv {

struct EngineConfig {
    int max_frames = 640;    // frames per CNN launch (one bucket graph per size class)
    int max_windows = 512;   // windows per sequence launch
    bool use_graphs = true;  // CUDA graphs per (slot, bucket); off for dumps / debugging
    int slots = 2;           // input/output buffer sets (pipelining in the scheduler)
};

// One CUDA context's worth of model state.  NOT thread-safe: one thread drives it (the API worker).
class Engine {
public:
    virtual ~Engine() = default;
    Precision precision() const { return prec_; }
    int max_frames() const { return cfg_.max_frames; }
    int max_windows() const { return cfg_.max_windows; }
    int slots() const { return cfg_.slots; }

    struct Slot {
        uint8_t* d_frames = nullptr;      // [max_frames][180*320*3] uint8 RGB crops
        float* d_tokens = nullptr;        // [max_frames][16*384] encode output (before positions)
        float* d_tok_stage = nullptr;     // [max_windows*20][16*384] host-provided tokens
        const float** d_tokptr = nullptr; // [max_windows*20] device pointers to 16x384 token blocks
        float* d_logits = nullptr;        // [max_windows]
    };
    const Slot& slot(int i) const { return slots_[size_t(i)]; }

    // Enqueue on `s`: tokens of slot.d_frames[0..F) -> slot.d_tokens; classify windows slot.d_tokptr[0..B*20) ->
    // slot.d_logits.  Padded to a bucket size when graphs are on (padding rows are computed and ignored).
    virtual void encode(int slot, int F, cudaStream_t s) = 0;
    virtual void classify(int slot, int B, cudaStream_t s) = 0;
    // Bucket sizes actually launched for F frames / B windows.
    int frame_bucket(int F) const;
    int window_bucket(int B) const;

    size_t weight_bytes() const { return weight_bytes_; }
    size_t activation_bytes() const { return act_bytes_; }
    size_t slot_bytes() const { return slot_bytes_; }
    // Debug: write intermediates of frame 0 / window 0 as raw float32 files into dir (graphs must be off).
    void set_dump_dir(const std::string& dir) { dump_dir_ = dir; }

protected:
    Precision prec_ = Precision::FP32;
    EngineConfig cfg_;
    std::vector<Slot> slots_;
    std::vector<int> fbuckets_, wbuckets_;
    size_t weight_bytes_ = 0, act_bytes_ = 0, slot_bytes_ = 0;
    std::string dump_dir_;
};

std::unique_ptr<Engine> make_engine(const HostModel& hm, Precision p, const EngineConfig& cfg);
const char* precision_name(Precision p);
bool parse_precision(const std::string& s, Precision* p);

}  // namespace shv
