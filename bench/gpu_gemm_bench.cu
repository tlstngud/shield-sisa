// gpu_gemm_bench: engine GEMM vs cuBLAS on the model's shapes, plus a bitwise check that every engine kernel
// variant (tile shape, pipeline depth, small-M GEMV) gives identical outputs (the basis of row independence).
//   gpu_gemm_bench [bf16|fp16|fp32] [iters]
#include "gemm.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace shv;

#define CB(x) do { cublasStatus_t s__ = (x); if (s__ != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublas %d line %d\n", int(s__), __LINE__); std::exit(1); } } while (0)

template <typename T> cudaDataType_t dt();
template <> cudaDataType_t dt<__nv_bfloat16>() { return CUDA_R_16BF; }
template <> cudaDataType_t dt<__half>() { return CUDA_R_16F; }
template <> cudaDataType_t dt<float>() { return CUDA_R_32F; }

template <typename T> __global__ void fill(T* p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        unsigned x = unsigned(i) * 2654435761u + seed;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = from_f<T>((float(x & 0xffff) / 65535.0f - 0.5f));
    }
}

template <typename T>
int run(int iters) {
    struct Shape { const char* name; int M, N, K, c_float, act, bias; };
    const Shape shapes[] = {
        {"b2.expand 320f", 320 * 14400, 64, 16, 0, ACT_RELU, 1},  {"b1.project 320f", 320 * 14400, 16, 16, 0, 0, 1},
        {"b3.expand 320f", 320 * 3600, 72, 24, 0, ACT_RELU, 1},   {"b3.project 320f", 320 * 3600, 24, 72, 0, 0, 1},
        {"b7.expand 320f", 320 * 920, 240, 40, 0, ACT_HSWISH, 1}, {"b12.expand 320f", 320 * 240, 672, 112, 0, ACT_HSWISH, 1},
        {"b12.project 320f", 320 * 240, 112, 672, 0, 0, 1},       {"b14.expand 320f", 320 * 60, 960, 160, 0, ACT_HSWISH, 1},
        {"gdn.in 32w", 32 * 321, 2316, 384, 1, 0, 0},             {"gdn.o 32w", 32 * 321, 384, 768, 1, 0, 0},
        {"ffn.up 32w", 32 * 321, 1920, 384, 0, 0, 0},             {"ffn.down 32w", 32 * 321, 384, 960, 1, 0, 0},
        {"gdn.in 1w", 321, 2316, 384, 1, 0, 0},                   {"gdn.in 512w", 512 * 321, 2316, 384, 1, 0, 0},
        {"lastrow.qc 7w", 7, 576, 384, 1, 0, 0},                  {"se.fc1 20f", 20, 240, 960, 1, ACT_RELU, 1},
        {"lastrow.ffn 100w", 100, 1920, 384, 1, 0, 0},
    };
    const bool half = sizeof(T) == 2;
    const std::vector<int> variants = half ? std::vector<int>{3, 5, 7, 8, 9} : std::vector<int>{1, 2};
    cublasHandle_t h;
    CB(cublasCreate(&h));
    if (!half) CB(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int mismatches = 0;
    for (const Shape& s : shapes) {
        if (!half && s.M < 64 && false) continue;
        T *A, *B;
        void* C;
        float* bias;
        const size_t cbytes = size_t(s.M) * s.N * (s.c_float ? 4 : sizeof(T));
        SHV_CUDA_CHECK(cudaMalloc(&A, size_t(s.M) * s.K * sizeof(T)));
        SHV_CUDA_CHECK(cudaMalloc(&B, size_t(s.N) * s.K * sizeof(T)));
        SHV_CUDA_CHECK(cudaMalloc(&C, size_t(s.M) * s.N * 4));
        SHV_CUDA_CHECK(cudaMalloc(&bias, size_t(s.N) * 4));
        fill<T><<<1024, 256>>>(A, size_t(s.M) * s.K, 1);
        fill<T><<<1024, 256>>>(B, size_t(s.N) * s.K, 2);
        fill<float><<<64, 256>>>(bias, size_t(s.N), 3);
        // fp32 operands: the engine never feeds storage-type outputs to fp32 GEMMs, so c_float only
        GemmParams p;
        p.M = s.M; p.N = s.N; p.K = s.K; p.A = A; p.lda = s.K; p.B = B; p.ldb = s.K; p.C = C; p.ldc = s.N;
        p.c_float = half ? s.c_float : 1;
        p.act = s.act;
        p.bias = s.bias ? bias : nullptr;
        auto time = [&](auto&& fn) {
            for (int i = 0; i < 2; ++i) fn();
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i) fn();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            return ms / iters;
        };
        gemm_force_kernel(-1);
        const float ours = time([&] { gemm<T>(p, nullptr); });
        SHV_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned char> ref(cbytes), got(cbytes);
        SHV_CUDA_CHECK(cudaMemcpy(ref.data(), C, cbytes, cudaMemcpyDeviceToHost));
        std::string vstr;
        for (int v : variants) {
            if (v == 1 && s.M > 20000) continue;  // GEMV only for small M (slow otherwise; still compared there)
            gemm_force_kernel(v);
            SHV_CUDA_CHECK(cudaMemset(C, 0xff, cbytes));
            const float t = time([&] { gemm<T>(p, nullptr); });
            SHV_CUDA_CHECK(cudaDeviceSynchronize());
            SHV_CUDA_CHECK(cudaMemcpy(got.data(), C, cbytes, cudaMemcpyDeviceToHost));
            const bool same = std::memcmp(ref.data(), got.data(), cbytes) == 0;
            mismatches += !same;
            char buf[64];
            std::snprintf(buf, sizeof buf, " v%d %.1fus %s", v, t * 1e3, same ? "==" : "DIFF");
            vstr += buf;
        }
        gemm_force_kernel(-1);
        const float one = 1.0f, zero = 0.0f;
        const cudaDataType_t ct = p.c_float ? CUDA_R_32F : dt<T>();
        const float cub = time([&] {
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.N, s.M, s.K, &one, B, dt<T>(), s.K, A, dt<T>(), s.K, &zero, C, ct, s.N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        });
        const double flop = 2.0 * s.M * double(s.N) * s.K;
        const double bytes = (double(s.M) * s.K + double(s.N) * s.K) * sizeof(T) + double(s.M) * s.N * (p.c_float ? 4 : sizeof(T));
        std::printf("%-17s M=%7d N=%5d K=%4d ours %8.1f us %6.1f TF %5.0f GB/s | cublas %8.1f us %6.1f TF | %s\n", s.name, s.M, s.N,
                    s.K, ours * 1e3, flop / ours / 1e9, bytes / ours / 1e6, cub * 1e3, flop / cub / 1e9, vstr.c_str());
        cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(bias);
    }
    cublasDestroy(h);
    std::printf("variant mismatches: %d\n", mismatches);
    return mismatches ? 1 : 0;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "bf16";
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    gemm_init_attributes();
    if (mode == "bf16") return run<__nv_bfloat16>(iters);
    if (mode == "fp16") return run<__half>(iters);
    return run<float>(iters);
}
