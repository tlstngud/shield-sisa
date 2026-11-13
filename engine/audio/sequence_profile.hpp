#pragma once
#include "core.hpp"
#include "sequence_cache.hpp"
namespace shield {
// Benchmark helper only; stages are GDN-A, GDN-B, final-row SISA, and head.
void sequence_profile(const Weights&, const float* tokens, float& logit, double stage_ms[4]);
}
