#include "precision.hpp"
#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace shield {
namespace {
thread_local ComputeMode mode=ComputeMode::Exact;
enum class Format {Fp16,Bf16,Int8};
enum class Region {Cnn,Sequence,Head};
struct Key {
    const float* weight; Format format;
    bool operator==(const Key& x)const{return weight==x.weight&&format==x.format;}
};
struct Hash {std::size_t operator()(const Key& x)const{return std::hash<const float*>{}(x.weight)^(std::size_t(x.format)<<2);}};
struct Packed {int in,out,pairs;Format format;std::vector<uint16_t> values;std::vector<int8_t> bytes;std::vector<float> scales;std::vector<int32_t> correction;};
std::unordered_map<Key,std::shared_ptr<const Packed>,Hash> packs;
std::unordered_map<const float*,Region> regions;
std::mutex mutex;
uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
float value(uint32_t x){float f;std::memcpy(&f,&x,4);return f;}
uint16_t to_bf16(float x) {
    uint32_t u=bits(x);
    // VCVTNEPS2BF16 treats FP32 subnormal inputs as signed zero regardless of
    // MXCSR. Match that behavior in the portable encoder and weight packer.
    if((u&0x7f800000u)==0)return uint16_t((u>>16)&0x8000u);
    if((u&0x7f800000u)==0x7f800000u&&(u&0x007fffffu)) return uint16_t((u>>16)|0x40u);
    return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
float from_bf16(uint16_t x){return value(uint32_t(x)<<16);}
uint16_t to_fp16(float x) {
    const uint32_t u=bits(x), sign=(u>>16)&0x8000u, raw_exp=(u>>23)&255u;
    uint32_t mant=u&0x7fffffu; int exponent=int(raw_exp)-127+15;
    if(raw_exp==255) return uint16_t(sign|0x7c00u|(mant?((mant>>13)|0x200u):0u));
    if(exponent>=31) return uint16_t(sign|0x7c00u);
    if(exponent<=0) {
        if(exponent < -10) return uint16_t(sign);
        mant|=0x800000u; const int shift=14-exponent;
        const uint32_t rounded=(mant+((1u<<(shift-1))-1u)+((mant>>shift)&1u))>>shift;
        return uint16_t(sign|rounded);
    }
    mant+=0xfffu+((mant>>13)&1u);
    if(mant&0x800000u){mant=0;++exponent;}
    if(exponent>=31) return uint16_t(sign|0x7c00u);
    return uint16_t(sign|(uint32_t(exponent)<<10)|(mant>>13));
}
float from_fp16(uint16_t h) {
    uint32_t sign=uint32_t(h&0x8000u)<<16, mant=h&0x3ffu, exponent=(h>>10)&31u;
    if(exponent==0) {
        if(!mant)return value(sign);
        int e=-14; while(!(mant&0x400u)){mant<<=1;--e;}
        return value(sign|(uint32_t(e+127)<<23)|((mant&0x3ffu)<<13));
    }
    if(exponent==31)return value(sign|0x7f800000u|(mant<<13));
    return value(sign|((exponent+112)<<23)|(mant<<13));
}
bool can_avx512(){return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("fma");}
bool can_avx2(){return __builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma");}
bool allowed(const char* isa) {
    const char* forced=std::getenv("SHIELD_KERNEL");
    return !forced||std::strcmp(forced,isa)==0||std::strcmp(forced,"compiler")==0;
}
thread_local std::vector<float> rounded;
thread_local std::vector<uint32_t> pairs;
thread_local std::vector<float> row_scales;
int nearest_even(float x) {
    const float lo=std::floor(x),fraction=x-lo;const int base=int(lo);
    return base+int(fraction>0.5f||(fraction==0.5f&&(base&1)));
}
__attribute__((target("avx512f")))
void quantize_int8_native(const float* a,int rows,int in) {
    const int count=(in+3)/4;pairs.resize(std::size_t(rows)*count);row_scales.resize(rows);
    for(int r=0;r<rows;++r) {
        float maximum=0;for(int q=0;q<in;++q)maximum=std::max(maximum,std::abs(a[r*in+q]));
        const float scale=maximum>0?maximum/127.0f:1.0f;row_scales[r]=scale;
        const __m512 inv=_mm512_set1_ps(1.0f/scale);uint8_t* dest=reinterpret_cast<uint8_t*>(pairs.data()+r*count);int q=0;
        for(;q+16<=in;q+=16) {
            __m512i v=_mm512_cvt_roundps_epi32(_mm512_mul_ps(_mm512_loadu_ps(a+r*in+q),inv),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
            v=_mm512_add_epi32(v,_mm512_set1_epi32(128));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dest+q),_mm512_cvtusepi32_epi8(v));
        }
        const float inverse=1.0f/scale;
        for(;q<in;++q)dest[q]=uint8_t(std::clamp(nearest_even(a[r*in+q]*inverse)+128,0,255));
        for(;q<count*4;++q)dest[q]=128;
    }
}
void quantize_int8(const float* a,int rows,int in) {
    if(__builtin_cpu_supports("avx512f")){quantize_int8_native(a,rows,in);return;}
    const int count=(in+3)/4;pairs.resize(std::size_t(rows)*count);row_scales.resize(rows);
    for(int r=0;r<rows;++r) {
        float maximum=0;for(int q=0;q<in;++q)maximum=std::max(maximum,std::abs(a[r*in+q]));
        const float scale=maximum>0?maximum/127.0f:1.0f;row_scales[r]=scale;const float inverse=1.0f/scale;
        uint8_t* dest=reinterpret_cast<uint8_t*>(pairs.data()+r*count);
        for(int q=0;q<count*4;++q)dest[q]=q<in?uint8_t(std::clamp(nearest_even(a[r*in+q]*inverse)+128,0,255)):128;
    }
}

__attribute__((target("avx2,f16c")))
void round_fp16_f16c(const float* a,float* b,std::size_t size) {
    std::size_t i=0;
    for(;i+8<=size;i+=8) {
        const __m128i h=_mm256_cvtps_ph(_mm256_loadu_ps(a+i),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        _mm256_storeu_ps(b+i,_mm256_cvtph_ps(h));
    }
    for(;i<size;++i)b[i]=from_fp16(to_fp16(a[i]));
}
void round_fp16(const float* a,std::size_t size) {
    rounded.resize(size);
    if(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("f16c")) round_fp16_f16c(a,rounded.data(),size);
    else for(std::size_t i=0;i<size;++i)rounded[i]=from_fp16(to_fp16(a[i]));
}
__attribute__((target("avx512f,avx512bf16")))
void round_bf16_native(const float* a,int rows,int in) {
    const int count=(in+1)/2;pairs.resize(std::size_t(rows)*count);
    for(int r=0;r<rows;++r) {
        uint16_t* dst=reinterpret_cast<uint16_t*>(pairs.data()+r*count);int q=0;
        for(;q+32<=in;q+=32) {
            const __m512bh v=_mm512_cvtne2ps_pbh(_mm512_loadu_ps(a+r*in+q+16),_mm512_loadu_ps(a+r*in+q));
            _mm512_storeu_si512(dst+q,(__m512i)v);
        }
        if(q+16<=in){const __m256bh v=_mm512_cvtneps_pbh(_mm512_loadu_ps(a+r*in+q));_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+q),(__m256i)v);q+=16;}
        if(q<in){const int tail=in-q;alignas(32) uint16_t last[16];
            const __m256bh v=_mm512_cvtneps_pbh(_mm512_maskz_loadu_ps((__mmask16)((1u<<tail)-1),a+r*in+q));
            _mm256_store_si256(reinterpret_cast<__m256i*>(last),(__m256i)v);std::memcpy(dst+q,last,std::size_t((tail+1)/2)*4);}
    }
}
void round_bf16_pairs(const float* a,int rows,int in) {
    if(__builtin_cpu_supports("avx512bf16")&&can_avx512()){round_bf16_native(a,rows,in);return;}
    const int count=(in+1)/2; pairs.resize(std::size_t(rows)*count);
    for(int r=0;r<rows;++r)for(int q=0;q<count;++q) {
        const uint32_t lo=to_bf16(a[r*in+q*2]);
        const uint32_t hi=q*2+1<in?to_bf16(a[r*in+q*2+1]):0;
        pairs[r*count+q]=lo|(hi<<16);
    }
}

__attribute__((target("avx512f,f16c,fma")))
void fp16_avx512_generic(const float* a,float* c,int m,int k,int n,const uint16_t* p,const float* bias) {
    for(int j=0;j<n;j+=32)for(int r=0;r<m;r+=6) {
        __m512 s[6][2];for(int z=0;z<6;++z)s[z][0]=s[z][1]=_mm512_setzero_ps();
        const uint16_t* w=p+(j/32)*k*32;const int nr=std::min(6,m-r);
        for(int q=0;q<k;++q) {
            const __m512 w0=_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+q*32)));
            const __m512 w1=_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+q*32+16)));
            for(int z=0;z<nr;++z) {const __m512 x=_mm512_set1_ps(a[(r+z)*k+q]);
                s[z][0]=_mm512_fmadd_ps(x,w0,s[z][0]);s[z][1]=_mm512_fmadd_ps(x,w1,s[z][1]);}
        }
        const int tail=std::min(32,n-j);const __mmask16 lo=(__mmask16)((1u<<std::min(16,tail))-1),hi=tail>16?(__mmask16)((1u<<(tail-16))-1):0;
        const __m512 b0=bias?_mm512_maskz_loadu_ps(lo,bias+j):_mm512_setzero_ps();
        const __m512 b1=bias&&tail>16?_mm512_maskz_loadu_ps(hi,bias+j+16):_mm512_setzero_ps();
        for(int z=0;z<nr;++z){_mm512_mask_storeu_ps(c+(r+z)*n+j,lo,_mm512_add_ps(s[z][0],b0));
            if(tail>16)_mm512_mask_storeu_ps(c+(r+z)*n+j+16,hi,_mm512_add_ps(s[z][1],b1));}
    }
}
__attribute__((target("avx512f,f16c,fma")))
void fp16_avx512(const float* a,float* c,int m,int k,int n,const uint16_t* p,const float* bias) {
    const int full=m/6*6;
    for(int j=0;j<n;j+=32)for(int r=0;r<full;r+=6) {
        __m512 s00=_mm512_setzero_ps(),s01=s00,s10=s00,s11=s00,s20=s00,s21=s00,s30=s00,s31=s00,s40=s00,s41=s00,s50=s00,s51=s00;
        const uint16_t* w=p+(j/32)*k*32;const float* x=a+r*k;
        for(int q=0;q<k;++q) {
            const __m512 w0=_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+q*32)));
            const __m512 w1=_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+q*32+16)));
#define UPDATE(Z) {const __m512 v=_mm512_set1_ps(x[(Z)*k+q]);s##Z##0=_mm512_fmadd_ps(v,w0,s##Z##0);s##Z##1=_mm512_fmadd_ps(v,w1,s##Z##1);}
            UPDATE(0) UPDATE(1) UPDATE(2) UPDATE(3) UPDATE(4) UPDATE(5)
#undef UPDATE
        }
        const int tail=std::min(32,n-j);const __mmask16 lo=(__mmask16)((1u<<std::min(16,tail))-1),hi=tail>16?(__mmask16)((1u<<(tail-16))-1):0;
        const __m512 b0=bias?_mm512_maskz_loadu_ps(lo,bias+j):_mm512_setzero_ps(),b1=bias&&tail>16?_mm512_maskz_loadu_ps(hi,bias+j+16):_mm512_setzero_ps();
#define STORE(Z) {_mm512_mask_storeu_ps(c+(r+(Z))*n+j,lo,_mm512_add_ps(s##Z##0,b0));if(tail>16)_mm512_mask_storeu_ps(c+(r+(Z))*n+j+16,hi,_mm512_add_ps(s##Z##1,b1));}
        STORE(0) STORE(1) STORE(2) STORE(3) STORE(4) STORE(5)
#undef STORE
    }
    if(full<m)fp16_avx512_generic(a+full*k,c+full*n,m-full,k,n,p,bias);
}
__attribute__((target("avx2,f16c,fma")))
void fp16_avx2(const float* a,float* c,int m,int k,int n,const uint16_t* p,const float* bias) {
    for(int j=0;j<n;j+=16)for(int r=0;r<m;r+=4) {
        __m256 s[4][2];for(int z=0;z<4;++z)s[z][0]=s[z][1]=_mm256_setzero_ps();
        const uint16_t* w=p+(j/32)*k*32+j%32;const int nr=std::min(4,m-r);
        for(int q=0;q<k;++q) {
            const __m256 w0=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+q*32)));
            const __m256 w1=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+q*32+8)));
            for(int z=0;z<nr;++z){const __m256 x=_mm256_set1_ps(a[(r+z)*k+q]);
                s[z][0]=_mm256_fmadd_ps(x,w0,s[z][0]);s[z][1]=_mm256_fmadd_ps(x,w1,s[z][1]);}
        }
        float temp[16];const int tail=std::min(16,n-j);
        for(int z=0;z<nr;++z){_mm256_storeu_ps(temp,s[z][0]);_mm256_storeu_ps(temp+8,s[z][1]);
            for(int t=0;t<tail;++t)c[(r+z)*n+j+t]=temp[t]+(bias?bias[j+t]:0.0f);}
    }
}
void fp16_scalar(const float* a,float* c,int m,int k,int n,const uint16_t* p,const float* bias) {
    for(int r=0;r<m;++r)for(int j=0;j<n;++j) {
        const uint16_t* w=p+(j/32)*k*32+j%32;float sum=0;
        for(int q=0;q<k;++q)sum=std::fma(a[r*k+q],from_fp16(w[q*32]),sum);
        c[r*n+j]=sum+(bias?bias[j]:0.0f);
    }
}
__attribute__((target("avx512f,avx512bf16")))
void bf16_native_generic(const uint32_t* a,float* c,int m,int count,int n,const uint16_t* p,const float* bias) {
    for(int j=0;j<n;j+=32)for(int r=0;r<m;r+=6) {
        __m512 s[6][2];for(int z=0;z<6;++z)s[z][0]=s[z][1]=_mm512_setzero_ps();
        const uint16_t* w=p+(j/32)*count*64;const int nr=std::min(6,m-r);
        for(int q=0;q<count;++q) {
            const __m512bh w0=(__m512bh)_mm512_loadu_si512(w+q*64);
            const __m512bh w1=(__m512bh)_mm512_loadu_si512(w+q*64+32);
            for(int z=0;z<nr;++z){const __m512bh x=(__m512bh)_mm512_set1_epi32(int(a[(r+z)*count+q]));
                s[z][0]=_mm512_dpbf16_ps(s[z][0],x,w0);s[z][1]=_mm512_dpbf16_ps(s[z][1],x,w1);}
        }
        const int tail=std::min(32,n-j);const __mmask16 lo=(__mmask16)((1u<<std::min(16,tail))-1),hi=tail>16?(__mmask16)((1u<<(tail-16))-1):0;
        const __m512 b0=bias?_mm512_maskz_loadu_ps(lo,bias+j):_mm512_setzero_ps();
        const __m512 b1=bias&&tail>16?_mm512_maskz_loadu_ps(hi,bias+j+16):_mm512_setzero_ps();
        for(int z=0;z<nr;++z){_mm512_mask_storeu_ps(c+(r+z)*n+j,lo,_mm512_add_ps(s[z][0],b0));
            if(tail>16)_mm512_mask_storeu_ps(c+(r+z)*n+j+16,hi,_mm512_add_ps(s[z][1],b1));}
    }
}
__attribute__((target("avx512f,avx512bf16")))
void bf16_native(const uint32_t* a,float* c,int m,int count,int n,const uint16_t* p,const float* bias) {
    const int full=m/6*6;
    for(int j=0;j<n;j+=32)for(int r=0;r<full;r+=6) {
        __m512 s00=_mm512_setzero_ps(),s01=s00,s10=s00,s11=s00,s20=s00,s21=s00,s30=s00,s31=s00,s40=s00,s41=s00,s50=s00,s51=s00;
        const uint16_t* w=p+(j/32)*count*64;const uint32_t* x=a+r*count;
        for(int q=0;q<count;++q) {
            const __m512bh w0=(__m512bh)_mm512_loadu_si512(w+q*64),w1=(__m512bh)_mm512_loadu_si512(w+q*64+32);
#define UPDATE(Z) {const __m512bh v=(__m512bh)_mm512_set1_epi32(int(x[(Z)*count+q]));s##Z##0=_mm512_dpbf16_ps(s##Z##0,v,w0);s##Z##1=_mm512_dpbf16_ps(s##Z##1,v,w1);}
            UPDATE(0) UPDATE(1) UPDATE(2) UPDATE(3) UPDATE(4) UPDATE(5)
#undef UPDATE
        }
        const int tail=std::min(32,n-j);const __mmask16 lo=(__mmask16)((1u<<std::min(16,tail))-1),hi=tail>16?(__mmask16)((1u<<(tail-16))-1):0;
        const __m512 b0=bias?_mm512_maskz_loadu_ps(lo,bias+j):_mm512_setzero_ps(),b1=bias&&tail>16?_mm512_maskz_loadu_ps(hi,bias+j+16):_mm512_setzero_ps();
#define STORE(Z) {_mm512_mask_storeu_ps(c+(r+(Z))*n+j,lo,_mm512_add_ps(s##Z##0,b0));if(tail>16)_mm512_mask_storeu_ps(c+(r+(Z))*n+j+16,hi,_mm512_add_ps(s##Z##1,b1));}
        STORE(0) STORE(1) STORE(2) STORE(3) STORE(4) STORE(5)
#undef STORE
    }
    if(full<m)bf16_native_generic(a+full*count,c+full*n,m-full,count,n,p,bias);
}
__attribute__((target("avx2,fma")))
void bf16_avx2(const uint32_t* a,float* c,int m,int count,int n,const uint16_t* p,const float* bias) {
    const __m256i mask=_mm256_set1_epi32(int(0xffff0000u));
    for(int j=0;j<n;j+=8)for(int r=0;r<m;r+=4) {
        __m256 s[4];for(auto& v:s)v=_mm256_setzero_ps();const int nr=std::min(4,m-r);
        const uint16_t* w=p+(j/32)*count*64+(j%32)*2;
        for(int q=0;q<count;++q) {
            const __m256i packed=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+q*64));
            const __m256 w0=_mm256_castsi256_ps(_mm256_slli_epi32(packed,16));
            const __m256 w1=_mm256_castsi256_ps(_mm256_and_si256(packed,mask));
            for(int z=0;z<nr;++z) {
                const uint32_t x=a[(r+z)*count+q];
                const __m256 x0=_mm256_set1_ps(value(x<<16)),x1=_mm256_set1_ps(value(x&0xffff0000u));
                // DPBF16 accumulates the high pair member first, then low.
                s[z]=_mm256_fmadd_ps(x1,w1,s[z]);s[z]=_mm256_fmadd_ps(x0,w0,s[z]);
            }
        }
        float temp[8];const int tail=std::min(8,n-j);
        for(int z=0;z<nr;++z){_mm256_storeu_ps(temp,s[z]);for(int t=0;t<tail;++t)c[(r+z)*n+j+t]=temp[t]+(bias?bias[j+t]:0.0f);}
    }
}
void bf16_scalar(const uint32_t* a,float* c,int m,int count,int n,const uint16_t* p,const float* bias) {
    for(int r=0;r<m;++r)for(int j=0;j<n;++j) {
        const uint16_t* w=p+(j/32)*count*64+(j%32)*2;float sum=0;
        for(int q=0;q<count;++q){const uint32_t x=a[r*count+q];
            sum=std::fma(from_bf16(uint16_t(x>>16)),from_bf16(w[q*64+1]),sum);
            sum=std::fma(from_bf16(uint16_t(x)),from_bf16(w[q*64]),sum);}
        c[r*n+j]=sum+(bias?bias[j]:0.0f);
    }
}
__attribute__((target("avx512f,avx512vnni")))
void int8_native(const uint32_t* a,float* c,int m,int count,int n,const Packed& packed,const float* bias) {
    const int full=m/6*6;
    for(int j=0;j<n;j+=32)for(int r=0;r<m;r+=6) {
        // The short final row tile is handled independently, preserving each
        // row's exact integer accumulation and its own activation scale.
        const int nr=std::min(6,m-r);const int8_t* w=packed.bytes.data()+(j/32)*count*128;
        __m512i s00=_mm512_setzero_si512(),s01=s00,s10=s00,s11=s00,s20=s00,s21=s00,s30=s00,s31=s00,s40=s00,s41=s00,s50=s00,s51=s00;
        for(int q=0;q<count;++q) {
            const __m512i w0=_mm512_loadu_si512(w+q*128),w1=_mm512_loadu_si512(w+q*128+64);
#define UPDATE(Z) {const __m512i v=_mm512_set1_epi32(int(a[(r+(Z))*count+q]));s##Z##0=_mm512_dpbusd_epi32(s##Z##0,v,w0);s##Z##1=_mm512_dpbusd_epi32(s##Z##1,v,w1);}
            UPDATE(0) if(nr>1){UPDATE(1)} if(nr>2){UPDATE(2)} if(nr>3){UPDATE(3)} if(nr>4){UPDATE(4)} if(nr>5){UPDATE(5)}
#undef UPDATE
        }
        const int tail=std::min(32,n-j);const __mmask16 lo=(__mmask16)((1u<<std::min(16,tail))-1),hi=tail>16?(__mmask16)((1u<<(tail-16))-1):0;
        const __m512i off0=_mm512_maskz_loadu_epi32(lo,packed.correction.data()+j),off1=tail>16?_mm512_maskz_loadu_epi32(hi,packed.correction.data()+j+16):_mm512_setzero_si512();
        const __m512 scales0=_mm512_maskz_loadu_ps(lo,packed.scales.data()+j),scales1=tail>16?_mm512_maskz_loadu_ps(hi,packed.scales.data()+j+16):_mm512_setzero_ps();
        const __m512 b0=bias?_mm512_maskz_loadu_ps(lo,bias+j):_mm512_setzero_ps(),b1=bias&&tail>16?_mm512_maskz_loadu_ps(hi,bias+j+16):_mm512_setzero_ps();
#define STORE(Z) {const __m512 scale=_mm512_set1_ps(row_scales[r+(Z)]);const __m512 y0=_mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(s##Z##0,off0)),scale),scales0),b0);_mm512_mask_storeu_ps(c+(r+(Z))*n+j,lo,y0);if(tail>16){const __m512 y1=_mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(s##Z##1,off1)),scale),scales1),b1);_mm512_mask_storeu_ps(c+(r+(Z))*n+j+16,hi,y1);}}
        STORE(0) if(nr>1){STORE(1)} if(nr>2){STORE(2)} if(nr>3){STORE(3)} if(nr>4){STORE(4)} if(nr>5){STORE(5)}
#undef STORE
    }
    (void)full;
}
__attribute__((target("avx2")))
void int8_avx2(const uint32_t* a,float* c,int m,int count,int n,const Packed& packed,const float* bias) {
    const __m256i masklow=_mm256_set1_epi8(0x7f),maskhigh=_mm256_set1_epi8(char(0x80)),ones=_mm256_set1_epi16(1);
    for(int j=0;j<n;j+=8)for(int r=0;r<m;++r) {
        __m256i sum=_mm256_setzero_si256();const int8_t* w=packed.bytes.data()+(j/32)*count*128+(j%32)*4;
        for(int q=0;q<count;++q) {
            const __m256i weights=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+q*128));
            const __m256i x=_mm256_set1_epi32(int(a[r*count+q]));
            const __m256i low=_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(x,masklow),weights),ones);
            const __m256i high=_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(x,maskhigh),weights),ones);
            sum=_mm256_add_epi32(sum,_mm256_add_epi32(low,high));
        }
        int32_t temp[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(temp),sum);const int tail=std::min(8,n-j);
        for(int t=0;t<tail;++t)c[r*n+j+t]=(float(temp[t]-packed.correction[j+t])*row_scales[r])*packed.scales[j+t]+(bias?bias[j+t]:0.0f);
    }
}
void int8_scalar(const uint32_t* a,float* c,int m,int count,int n,const Packed& packed,const float* bias) {
    for(int r=0;r<m;++r)for(int j=0;j<n;++j) {
        const uint8_t* x=reinterpret_cast<const uint8_t*>(a+r*count);const int8_t* w=packed.bytes.data()+(j/32)*count*128+(j%32)*4;int32_t sum=0;
        for(int q=0;q<count;++q)for(int z=0;z<4;++z)sum+=int32_t(x[q*4+z])*int32_t(w[q*128+z]);
        c[r*n+j]=(float(sum-packed.correction[j])*row_scales[r])*packed.scales[j]+(bias?bias[j]:0.0f);
    }
}
std::shared_ptr<const Packed> get_packed(const float* w,int k,int n,Format format) {
    const Key key{w,format};const auto found=packs.find(key);if(found!=packs.end()) {
        if(found->second->in!=k||found->second->out!=n)throw std::runtime_error("approximate linear: incompatible weight shape");
        return found->second;
    }
    auto result=std::make_shared<Packed>();result->in=k;result->out=n;result->format=format;result->pairs=(k+1)/2;
    if(format==Format::Fp16) {
        result->values.assign(((n+31)/32)*k*32,0);
        for(int j=0;j<n;++j)for(int q=0;q<k;++q)result->values[(j/32)*k*32+q*32+j%32]=to_fp16(w[j*k+q]);
    } else if(format==Format::Bf16) {
        result->values.assign(((n+31)/32)*result->pairs*64,0);
        for(int j=0;j<n;++j)for(int q=0;q<k;++q)result->values[(j/32)*result->pairs*64+(q/2)*64+(j%32)*2+q%2]=to_bf16(w[j*k+q]);
    } else {
        result->pairs=(k+3)/4;result->bytes.assign(((n+31)/32)*result->pairs*128,0);result->scales.resize(n);result->correction.assign(n,0);
        for(int j=0;j<n;++j) {
            float maximum=0;for(int q=0;q<k;++q)maximum=std::max(maximum,std::abs(w[j*k+q]));
            const float scale=maximum>0?maximum/127.0f:1.0f;result->scales[j]=scale;const float inverse=1.0f/scale;int32_t sum=0;
            for(int q=0;q<k;++q){const int8_t v=int8_t(std::clamp(nearest_even(w[j*k+q]*inverse),-127,127));sum+=v;
                result->bytes[(j/32)*result->pairs*128+(q/4)*128+(j%32)*4+q%4]=v;}
            result->correction[j]=sum*128;
        }
    }
    packs.emplace(key,result);return result;
}
} // namespace

ComputeMode get_compute_mode()noexcept{return mode;}
ComputeMode set_compute_mode(ComputeMode next)noexcept{const auto old=mode;mode=next;return old;}
std::uint64_t execution_mode_key()noexcept{return std::uint64_t(mode);}
const char* compute_revision()noexcept{return "linear16-int8-v1";}
const char* compute_mode_name(ComputeMode x)noexcept {
    switch(x){case ComputeMode::Exact:return "exact";case ComputeMode::Fp16:return "fp16-linear";
    case ComputeMode::Bf16:return "bf16-linear";case ComputeMode::Fp16Cnn:return "fp16-cnn";case ComputeMode::Bf16Cnn:return "bf16-cnn";case ComputeMode::Int8Cnn:return "int8-cnn";}
    return "invalid";
}
bool compute_mode_accelerated(ComputeMode x)noexcept {
    if(x==ComputeMode::Exact)return true;
    if(x==ComputeMode::Int8Cnn)return can_avx512()&&__builtin_cpu_supports("avx512vnni");
    if(x==ComputeMode::Fp16||x==ComputeMode::Fp16Cnn)return __builtin_cpu_supports("f16c")&&(can_avx512()||can_avx2());
    return __builtin_cpu_supports("avx512bf16")&&can_avx512();
}
void register_compute_weights(const Weights& w) {
    std::lock_guard<std::mutex> lock(mutex);
    for(const auto& t:w.tensors){const auto& name=t.first;
        const Region region=name.rfind("cnn.",0)==0?Region::Cnn:name.rfind("head.",0)==0||name.rfind("head_norm.",0)==0?Region::Head:Region::Sequence;
        regions[t.second.data.data()]=region;}
}
namespace detail {
bool approximate_linear(const float* a,float* c,int m,int k,int n,const float* w,const float* b) {
    if(mode==ComputeMode::Exact)return false;
    const bool cnn_only=mode==ComputeMode::Fp16Cnn||mode==ComputeMode::Bf16Cnn||mode==ComputeMode::Int8Cnn;
    const Format format=mode==ComputeMode::Fp16||mode==ComputeMode::Fp16Cnn?Format::Fp16:mode==ComputeMode::Int8Cnn?Format::Int8:Format::Bf16;
    // Small input/output projections have too little arithmetic to amortize
    // dynamic activation quantization. Keep stem and frequency attention FP32.
    if(format==Format::Int8&&(k<32||n<32))return false;
    std::shared_ptr<const Packed> p;
    {std::lock_guard<std::mutex> lock(mutex);const auto region=regions.find(w);
        if(region!=regions.end()&&region->second==Region::Head)return false;
        if(cnn_only&&(region==regions.end()||region->second!=Region::Cnn))return false;
        p=get_packed(w,k,n,format);}
    if(format==Format::Fp16) {
        round_fp16(a,std::size_t(m)*k);
        if(can_avx512()&&__builtin_cpu_supports("f16c")&&allowed("avx512")) fp16_avx512(rounded.data(),c,m,k,n,p->values.data(),b);
        else if(can_avx2()&&__builtin_cpu_supports("f16c")&&(allowed("avx2")||allowed("avx512"))) fp16_avx2(rounded.data(),c,m,k,n,p->values.data(),b);
        else fp16_scalar(rounded.data(),c,m,k,n,p->values.data(),b);
    } else if(format==Format::Bf16) {
        round_bf16_pairs(a,m,k);
        if(can_avx512()&&__builtin_cpu_supports("avx512bf16")&&allowed("avx512")) bf16_native(pairs.data(),c,m,p->pairs,n,p->values.data(),b);
        else if(can_avx2()&&(allowed("avx2")||allowed("avx512"))) bf16_avx2(pairs.data(),c,m,p->pairs,n,p->values.data(),b);
        else bf16_scalar(pairs.data(),c,m,p->pairs,n,p->values.data(),b);
    } else {
        quantize_int8(a,m,k);
        if(can_avx512()&&__builtin_cpu_supports("avx512vnni")&&allowed("avx512"))int8_native(pairs.data(),c,m,p->pairs,n,*p,b);
        else if(__builtin_cpu_supports("avx2")&&(allowed("avx2")||allowed("avx512")))int8_avx2(pairs.data(),c,m,p->pairs,n,*p,b);
        else int8_scalar(pairs.data(),c,m,p->pairs,n,*p,b);
    }
    return true;
}
void release_precision_weights(const Weights& w){std::lock_guard<std::mutex> lock(mutex);for(const auto& t:w.tensors){const float* p=t.second.data.data();regions.erase(p);packs.erase(Key{p,Format::Fp16});packs.erase(Key{p,Format::Bf16});packs.erase(Key{p,Format::Int8});}}
std::size_t precision_weight_bytes(){std::lock_guard<std::mutex> lock(mutex);std::size_t count=0;for(const auto& p:packs)count+=p.second->values.size()*sizeof(uint16_t)+p.second->bytes.size()+p.second->scales.size()*sizeof(float)+p.second->correction.size()*sizeof(int32_t);return count;}
}
} // namespace shield
