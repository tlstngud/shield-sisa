// GEMM dispatch: tile shape from (M, N) only. All shapes accumulate k in the same order (see gemm.cuh).
#include "gemm.cuh"

namespace shv {
namespace {
int g_sm_count = 84;
// Test hook (bench/gpu_gemm_bench): force one kernel family / tile shape to show they give identical bits.
int g_force = -1;

template <int BM, int BN, int TM, int TN>
void launch_simt(const GemmParams& p, cudaStream_t s) {
    const dim3 grid(ceil_div(p.M, BM), ceil_div(p.N, BN));
    constexpr int threads = (BM / TM) * (BN / TN);
    if (p.a_scale) gemm_simt<BM, BN, TM, TN, true><<<grid, threads, 0, s>>>(p);
    else gemm_simt<BM, BN, TM, TN, false><<<grid, threads, 0, s>>>(p);
}

template <typename T, int BM, int BN, int WM, int WN, int ST>
void launch_mma(const GemmParams& p, cudaStream_t s) {
    const dim3 grid(ceil_div(p.M, BM), ceil_div(p.N, BN));
    constexpr int threads = (BM / WM) * (BN / WN) * 32;
    constexpr int smem = mma_smem_bytes<BM, BN, ST>();
    if (p.a_scale) gemm_mma<T, BM, BN, WM, WN, ST, true><<<grid, threads, smem, s>>>(p);
    else gemm_mma<T, BM, BN, WM, WN, ST, false><<<grid, threads, smem, s>>>(p);
}

template <typename T, int BM, int BN, int WM, int WN, int ST>
void set_attr() {
    constexpr int smem = mma_smem_bytes<BM, BN, ST>();
    SHV_CUDA_CHECK(cudaFuncSetAttribute(gemm_mma<T, BM, BN, WM, WN, ST, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    SHV_CUDA_CHECK(cudaFuncSetAttribute(gemm_mma<T, BM, BN, WM, WN, ST, false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
}
template <typename T, int BM, int BN, int WM, int WN>
void launch_mma_k(const GemmParams& p, cudaStream_t s) {
    // shallow pipeline (less shared memory, more resident CTAs) when there are only one or two k tiles
    if (ceil_div(p.K, MMA_BK) <= 2) launch_mma<T, BM, BN, WM, WN, 2>(p, s);
    else launch_mma<T, BM, BN, WM, WN, 3>(p, s);
}
template <typename T, int BM, int BN, int WM, int WN> void set_attr2() {
    set_attr<T, BM, BN, WM, WN, 2>();
    set_attr<T, BM, BN, WM, WN, 3>();
}
template <typename T> void set_attrs() {
    set_attr2<T, 128, 16, 32, 16>();
    set_attr2<T, 128, 32, 32, 32>();
    set_attr2<T, 128, 64, 64, 32>();
    set_attr2<T, 128, 128, 64, 32>();
    set_attr<T, 128, 128, 64, 32, 4>();
    set_attr2<T, 64, 64, 32, 32>();
    set_attr2<T, 32, 64, 16, 32>();
    set_attr<T, 128, 256, 64, 64, 2>();
    set_attr<T, 256, 128, 64, 64, 2>();
}

void check(const GemmParams& p, int elem) {
    if (p.M <= 0 || p.N <= 0 || p.K <= 0) throw std::runtime_error("gemm: empty shape");
    if (p.K % 8 || p.lda % 8 || p.ldb % 8) throw std::runtime_error("gemm: K/lda/ldb must be multiples of 8");
    if ((reinterpret_cast<uintptr_t>(p.A) | reinterpret_cast<uintptr_t>(p.B)) % 16)
        throw std::runtime_error("gemm: operands must be 16-byte aligned");
    if (p.swiglu && (p.N & 1)) throw std::runtime_error("gemm: swiglu needs even N");
    (void)elem;
}
}  // namespace

void gemm_force_kernel(int k) { g_force = k; }
void gemm_init_attributes() {
    int dev = 0;
    SHV_CUDA_CHECK(cudaGetDevice(&dev));
    SHV_CUDA_CHECK(cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev));
    set_attrs<__nv_bfloat16>();
    set_attrs<__half>();
}

template <> void gemm<float>(const GemmParams& p, cudaStream_t s) {
    check(p, 4);
    const bool small = g_force >= 0 ? g_force == 1 : p.M <= 32;
    if (small) {
        const long long n = (long long)p.M * ((p.N + 1) / 2);
        if (p.a_scale) gemv_seq<true><<<ceil_div(n, 128), 128, 0, s>>>(p);
        else gemv_seq<false><<<ceil_div(n, 128), 128, 0, s>>>(p);
    } else if (g_force == 2) launch_simt<64, 64, 8, 8>(p, s);
    else if (p.N <= 32) launch_simt<128, 32, 8, 4>(p, s);
    else if (p.N <= 64) launch_simt<128, 64, 8, 8>(p, s);
    else if ((long long)ceil_div(p.M, 128) * ceil_div(p.N, 128) < 2LL * g_sm_count) launch_simt<64, 64, 8, 8>(p, s);
    else launch_simt<128, 128, 8, 8>(p, s);
    SHV_CUDA_CHECK(cudaGetLastError());
}

template <typename T> static void gemm_half(const GemmParams& p, cudaStream_t s) {
    check(p, 2);
    if (g_force == 3) { launch_mma_k<T, 32, 64, 16, 32>(p, s); SHV_CUDA_CHECK(cudaGetLastError()); return; }
    if (g_force == 4) { launch_mma_k<T, 128, 128, 64, 32>(p, s); SHV_CUDA_CHECK(cudaGetLastError()); return; }
    if (g_force == 5) { launch_mma<T, 128, 128, 64, 32, 2>(p, s); SHV_CUDA_CHECK(cudaGetLastError()); return; }
    if (g_force == 6) { launch_mma<T, 128, 128, 64, 32, 4>(p, s); SHV_CUDA_CHECK(cudaGetLastError()); return; }
    if (g_force == 7) { launch_mma_k<T, 64, 64, 32, 32>(p, s); SHV_CUDA_CHECK(cudaGetLastError()); return; }
    if (g_force == 8) { launch_mma<T, 128, 256, 64, 64, 2>(p, s); SHV_CUDA_CHECK(cudaGetLastError()); return; }
    if (g_force == 9) { launch_mma<T, 256, 128, 64, 64, 2>(p, s); SHV_CUDA_CHECK(cudaGetLastError()); return; }
    if (p.N <= 16) launch_mma_k<T, 128, 16, 32, 16>(p, s);
    else if (p.N <= 32) launch_mma_k<T, 128, 32, 32, 32>(p, s);
    else if (p.N <= 64) { if (p.K <= 32) launch_mma_k<T, 64, 64, 32, 32>(p, s); else launch_mma_k<T, 128, 64, 64, 32>(p, s); }
    else if ((long long)ceil_div(p.M, 64) * ceil_div(p.N, 64) < 2LL * g_sm_count) launch_mma_k<T, 32, 64, 16, 32>(p, s);
    else if ((long long)ceil_div(p.M, 128) * ceil_div(p.N, 128) < 2LL * g_sm_count) launch_mma_k<T, 64, 64, 32, 32>(p, s);
    else launch_mma<T, 128, 128, 64, 32, 2>(p, s);  // 2 stages: 41 KB smem, 2 CTAs per SM (measured faster than 3-4)
    SHV_CUDA_CHECK(cudaGetLastError());
}
template <> void gemm<__nv_bfloat16>(const GemmParams& p, cudaStream_t s) { gemm_half<__nv_bfloat16>(p, s); }
template <> void gemm<__half>(const GemmParams& p, cudaStream_t s) { gemm_half<__half>(p, s); }

}  // namespace shv
