#include "core.hpp"
#include "precision.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace shield {
namespace {
struct Packed { int in, out; std::vector<float> data; };
std::unordered_map<const float*,std::shared_ptr<Packed>> packs;
std::mutex pack_mutex;
using Kernel=void(*)(const float*,float*,int,int,int,const float*,const float*);
void scalar(const float* a,float* c,int m,int k,int n,const float* p,const float* bias) {
    for(int r=0;r<m;++r) for(int j=0;j<n;++j) {
        float s=0; const float* w=p+(j/32)*k*32+j%32;
        for(int q=0;q<k;++q) s=std::fma(a[r*k+q],w[q*32],s);
        c[r*n+j]=bias?s+bias[j]:s;
    }
}
__attribute__((target("avx512f,fma")))
void avx512_generic(const float* a,float* c,int m,int k,int n,const float* p,const float* bias) {
    for(int j=0;j<n;j+=32) for(int r=0;r<m;r+=6) {
        __m512 s[6][2]; for(int z=0;z<6;++z) s[z][0]=s[z][1]=_mm512_setzero_ps();
        const float* w=p+(j/32)*k*32;
        int nr=std::min(6,m-r);
        for(int q=0;q<k;++q) {
            __m512 w0=_mm512_loadu_ps(w+q*32),w1=_mm512_loadu_ps(w+q*32+16);
            for(int z=0;z<nr;++z) {
                __m512 x=_mm512_set1_ps(a[(r+z)*k+q]);
                s[z][0]=_mm512_fmadd_ps(x,w0,s[z][0]);
                s[z][1]=_mm512_fmadd_ps(x,w1,s[z][1]);
            }
        }
        int tail=std::min(32,n-j);
        for(int z=0;z<nr;++z) {
            if(bias) {s[z][0]=_mm512_add_ps(s[z][0],_mm512_maskz_loadu_ps((__mmask16)((1u<<std::min(16,tail))-1),bias+j));
                if(tail>16) s[z][1]=_mm512_add_ps(s[z][1],_mm512_maskz_loadu_ps((__mmask16)((1u<<(tail-16))-1),bias+j+16));}
            _mm512_mask_storeu_ps(c+(r+z)*n+j,(__mmask16)((1u<<std::min(16,tail))-1),s[z][0]);
            if(tail>16) _mm512_mask_storeu_ps(c+(r+z)*n+j+16,(__mmask16)((1u<<(tail-16))-1),s[z][1]);
        }
    }
}
__attribute__((target("avx512f,fma")))
void avx512(const float* a,float* c,int m,int k,int n,const float* p,const float* bias) {
    int full=m/6*6;
    for(int j=0;j<n;j+=32)for(int r=0;r<full;r+=6){
        __m512 s00=_mm512_setzero_ps(),s01=s00,s10=s00,s11=s00,s20=s00,s21=s00,s30=s00,s31=s00,s40=s00,s41=s00,s50=s00,s51=s00;
        const float* w=p+(j/32)*k*32;const float* x=a+r*k;
        for(int q=0;q<k;++q){__m512 w0=_mm512_loadu_ps(w+q*32),w1=_mm512_loadu_ps(w+q*32+16);
#define UPDATE(Z) {__m512 v=_mm512_set1_ps(x[(Z)*k+q]);s##Z##0=_mm512_fmadd_ps(v,w0,s##Z##0);s##Z##1=_mm512_fmadd_ps(v,w1,s##Z##1);}
            UPDATE(0) UPDATE(1) UPDATE(2) UPDATE(3) UPDATE(4) UPDATE(5)
#undef UPDATE
        }
        int tail=std::min(32,n-j);__mmask16 lo=(__mmask16)((1u<<std::min(16,tail))-1),hi=tail>16?(__mmask16)((1u<<(tail-16))-1):0;
        __m512 b0=bias?_mm512_maskz_loadu_ps(lo,bias+j):_mm512_setzero_ps(),b1=(bias&&tail>16)?_mm512_maskz_loadu_ps(hi,bias+j+16):_mm512_setzero_ps();
#define STORE(Z) {_mm512_mask_storeu_ps(c+(r+(Z))*n+j,lo,_mm512_add_ps(s##Z##0,b0));if(tail>16)_mm512_mask_storeu_ps(c+(r+(Z))*n+j+16,hi,_mm512_add_ps(s##Z##1,b1));}
        STORE(0) STORE(1) STORE(2) STORE(3) STORE(4) STORE(5)
#undef STORE
    }
    if(full<m)avx512_generic(a+full*k,c+full*n,m-full,k,n,p,bias);
}
__attribute__((target("avx2,fma")))
void avx2(const float* a,float* c,int m,int k,int n,const float* p,const float* bias) {
    for(int j=0;j<n;j+=16) for(int r=0;r<m;r+=4) {
        __m256 s[4][2];for(int z=0;z<4;++z)s[z][0]=s[z][1]=_mm256_setzero_ps();
        const float* w=p+(j/32)*k*32+j%32; int nr=std::min(4,m-r);
        for(int q=0;q<k;++q) {
            __m256 w0=_mm256_loadu_ps(w+q*32),w1=_mm256_loadu_ps(w+q*32+8);
            for(int z=0;z<nr;++z) {__m256 x=_mm256_set1_ps(a[(r+z)*k+q]); s[z][0]=_mm256_fmadd_ps(x,w0,s[z][0]);s[z][1]=_mm256_fmadd_ps(x,w1,s[z][1]);}
        }
        alignas(32) float tmp[16]; int tail=std::min(16,n-j);
        for(int z=0;z<nr;++z) {_mm256_store_ps(tmp,s[z][0]);_mm256_store_ps(tmp+8,s[z][1]);for(int t=0;t<tail;++t)c[(r+z)*n+j+t]=tmp[t]+(bias?bias[j+t]:0.f);}
    }
}
// Compiler-vectorized comparison kernel. It uses the same packed weights and
// reduction order as the intrinsics path, making dispatch comparisons useful.
__attribute__((target("avx512f,fma")))
void compiler_kernel(const float* a,float* c,int m,int k,int n,const float* p,const float* bias) {
    for(int j=0;j<n;j+=32)for(int r=0;r<m;r+=4){
        alignas(64) float acc[4][32]{};int nr=std::min(4,m-r),nc=std::min(32,n-j);
        const float* w=p+(j/32)*k*32;
        for(int q=0;q<k;++q)for(int z=0;z<nr;++z){float x=a[(r+z)*k+q];
            for(int v=0;v<32;++v)acc[z][v]=std::fma(x,w[q*32+v],acc[z][v]);}
        for(int z=0;z<nr;++z)for(int v=0;v<nc;++v)c[(r+z)*n+j+v]=acc[z][v]+(bias?bias[j+v]:0.f);
    }
}
struct Dispatch {Kernel fn;const char* name;};
Dispatch dispatch() {
    const char* forced=std::getenv("SHIELD_KERNEL");
    if(forced&&std::strcmp(forced,"scalar")==0)return {scalar,"scalar"};
    __builtin_cpu_init();
    if(forced&&std::strcmp(forced,"compiler")==0&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("fma"))return{compiler_kernel,"compiler"};
    if((!forced||std::strcmp(forced,"avx512")==0)&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("fma"))return{avx512,"avx512"};
    if((!forced||std::strcmp(forced,"avx2")==0)&&__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma"))return{avx2,"avx2"};
    return{scalar,"scalar"};
}
const Dispatch selected=dispatch();
}
void linear(const float* a,float* c,int m,int k,int n,const float* w,const float* b) {
    if(detail::approximate_linear(a,c,m,k,n,w,b))return;
    std::shared_ptr<Packed> p;
    {std::lock_guard<std::mutex> lock(pack_mutex);auto it=packs.find(w);if(it!=packs.end())p=it->second;
        else {p=std::make_shared<Packed>();p->in=k;p->out=n;p->data.assign(((n+31)/32)*k*32,0.f);
            for(int j=0;j<n;++j)for(int q=0;q<k;++q)p->data[(j/32)*k*32+q*32+j%32]=w[j*k+q];packs.emplace(w,p);}}
    if(p->in!=k||p->out!=n)throw std::runtime_error("linear: weight pointer reused with incompatible shape");
    selected.fn(a,c,m,k,n,p->data.data(),b);
}
const char* kernel_name(){return selected.name;}
void release_kernel_weights(const Weights& w){{std::lock_guard<std::mutex> lock(pack_mutex);for(const auto& item:w.tensors)packs.erase(item.second.data.data());}detail::release_precision_weights(w);}
std::size_t packed_weight_bytes(){std::size_t n=0;{std::lock_guard<std::mutex> lock(pack_mutex);for(auto& p:packs)n+=p.second->data.size()*sizeof(float);}return n+detail::precision_weight_bytes();}
}


namespace shield {
void batchnorm_activate(float* x,int rows,int channels,const float* scale,const float* shift,bool act){
    for(int r=0;r<rows;++r)for(int c=0;c<channels;++c){
        float y=x[r*channels+c]*scale[c]+shift[c];
        x[r*channels+c]=act?y/(1.f+std::exp(-y)):y;
    }
}
}
