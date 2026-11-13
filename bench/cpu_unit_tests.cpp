// Unit tests of the CPU engine kernels (built against isa_impl.hpp directly, AVX-512 level).
//   - vexp / vsilu / vhswish accuracy vs double precision (max ulp, mean signed error)
//   - GEMM accuracy vs double for K = 384, 768, 960 and bit-identity across tile shapes (M tails)
// Build: g++ -O2 -std=c++17 -I engine/include bench/cpu_unit_tests.cpp -L build/cpu -lshield_v_cpu
#define SHV_LEVEL 3
#include "../engine/model/isa_impl.hpp"

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

static double ulp_err(float got, double want) {
    if (want == 0) return got == 0 ? 0 : 1e9;
    const double u = std::ldexp(1.0, std::ilogb((float)want) - 23);
    return (got - want) / u;
}

int main() {
    std::mt19937 rng(1);
    // exp on [-30, 30]
    {
        double maxu = 0, sum = 0; long n = 0;
        for (int i = 0; i < 2000000; i += 16) {
            alignas(64) float x[16], y[16];
            for (int l = 0; l < 16; ++l) x[l] = -30.0f + 60.0f * (float)(i + l) / 2000000.0f;
            vstore(y, vexp(vload(x)));
            for (int l = 0; l < 16; ++l) { double e = ulp_err(y[l], std::exp((double)x[l])); maxu = std::fmax(maxu, std::fabs(e)); sum += e; ++n; }
        }
        std::printf("vexp   [-30,30]: max %.3f ulp, mean signed %.4f ulp\n", maxu, sum / n);
        maxu = 0; sum = 0; n = 0;
        for (int i = 0; i < 2000000; i += 16) {
            alignas(64) float x[16], y[16];
            for (int l = 0; l < 16; ++l) x[l] = -30.0f + 60.0f * (float)(i + l) / 2000000.0f;
            for (int l = 0; l < 16; ++l) y[l] = expf(x[l]);
            for (int l = 0; l < 16; ++l) { double e = ulp_err(y[l], std::exp((double)x[l])); maxu = std::fmax(maxu, std::fabs(e)); sum += e; ++n; }
        }
        std::printf("expf   [-30,30]: max %.3f ulp, mean signed %.4f ulp (glibc)\n", maxu, sum / n);
    }
    // silu on [-20, 20]
    {
        double maxu = 0, sum = 0, maxa = 0; long n = 0;
        for (int i = 0; i < 2000000; i += 16) {
            alignas(64) float x[16], y[16];
            for (int l = 0; l < 16; ++l) x[l] = -20.0f + 40.0f * (float)(i + l) / 2000000.0f;
            vstore(y, vsilu(vload(x)));
            for (int l = 0; l < 16; ++l) {
                double want = x[l] / (1.0 + std::exp(-(double)x[l]));
                double e = ulp_err(y[l], want); maxu = std::fmax(maxu, std::fabs(e)); sum += e; ++n;
                maxa = std::fmax(maxa, std::fabs(y[l] - want));
            }
        }
        std::printf("vsilu  [-20,20]: max %.3f ulp, mean signed %.4f ulp, max abs %.3e\n", maxu, sum / n, maxa);
    }
    // GEMM accuracy vs double and bit-identity across M
    for (int K : {384, 768, 960}) {
        const int N = 64, M = 37;
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> A((size_t)M * K), W((size_t)N * K), Wp((size_t)N * K), C((size_t)M * N), C1(N);
        for (float& v : A) v = nd(rng);
        for (float& v : W) v = nd(rng) * 0.05f;
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) Wp[((size_t)(n / 16) * K + k) * 16 + n % 16] = W[(size_t)n * K + k];
        shv::Linear L{};
        L.wp = Wp.data();
        L.K = K;
        L.N = N;
        L.Np = N;
        L.fmt = shv::FMT_F32;
        gemm(A.data(), K, M, &L, 0, N / 16, C.data(), N, shv::EPI_STORE, nullptr, 0);
        double maxe = 0, sum2 = 0, mean = 0;
        bool same = true;
        for (int m = 0; m < M; ++m) {
            gemm(A.data() + (size_t)m * K, K, 1, &L, 0, N / 16, C1.data(), N, shv::EPI_STORE, nullptr, 0);
            for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int k = 0; k < K; ++k) ref += (double)A[(size_t)m * K + k] * W[(size_t)n * K + k];
                double e = C[(size_t)m * N + n] - ref;
                maxe = std::fmax(maxe, std::fabs(e)); sum2 += e * e; mean += e;
                same &= C1[n] == C[(size_t)m * N + n];
            }
        }
        std::printf("gemm K=%d: max abs err %.3e, rms %.3e, mean %.2e, single-row == batched: %s\n", K, maxe,
                    std::sqrt(sum2 / (M * N)), mean / (M * N), same ? "yes" : "NO");
    }
    return 0;
}
