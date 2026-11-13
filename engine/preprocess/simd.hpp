// Per-function ISA targets (the library is compiled for baseline x86-64; kernels opt in and are selected at
// run time by shvp::isa()).  -ffp-contract=off is set for the whole library so no FMA contraction happens
// inside these functions either.
#pragma once
#if defined(__x86_64__)
#include <immintrin.h>
#define SHV_TARGET_AVX2 __attribute__((target("avx2,fma,bmi,bmi2,popcnt")))
#define SHV_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx2,fma,bmi,bmi2,popcnt")))
#else
#define SHV_TARGET_AVX2
#define SHV_TARGET_AVX512
#endif
