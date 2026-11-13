#include "frontend.hpp"
#include "precision.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

extern "C" __m512 _ZGVeN16v_expf(__m512);
extern "C" __m256 _ZGVdN8v_expf(__m256);
namespace shield {
namespace {
using BNKernel=void(*)(float*,int,int,const float*,const float*,bool);
using DWKernel=void(*)(const float*,float*,int,int,int,int,int,const float*);
__attribute__((target("avx512f")))
void bn512(float* x,int rows,int channels,const float* scale,const float* shift,bool act){
    for(int r=0;r<rows;++r)for(int c=0;c<channels;c+=16){__mmask16 mask=(__mmask16)((1u<<std::min(16,channels-c))-1);
        __m512 y=_mm512_add_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(mask,x+r*channels+c),_mm512_maskz_loadu_ps(mask,scale+c)),_mm512_maskz_loadu_ps(mask,shift+c));
        if(act)y=_mm512_div_ps(y,_mm512_add_ps(_mm512_set1_ps(1.f),_ZGVeN16v_expf(_mm512_sub_ps(_mm512_setzero_ps(),y))));
        _mm512_mask_storeu_ps(x+r*channels+c,mask,y);}
}
__attribute__((target("avx2")))
void bn256(float* x,int rows,int channels,const float* scale,const float* shift,bool act){
    for(int r=0;r<rows;++r){int c=0;for(;c+8<=channels;c+=8){
        __m256 y=_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x+r*channels+c),_mm256_loadu_ps(scale+c)),_mm256_loadu_ps(shift+c));
        if(act)y=_mm256_div_ps(y,_mm256_add_ps(_mm256_set1_ps(1.f),_ZGVdN8v_expf(_mm256_sub_ps(_mm256_setzero_ps(),y))));
        _mm256_storeu_ps(x+r*channels+c,y);}
        for(;c<channels;++c){float y=x[r*channels+c]*scale[c]+shift[c];x[r*channels+c]=act?y/(1.f+std::exp(-y)):y;}}
}
void dw_compiler(const float* x,float* y,int time,int freq,int channels,int sf,int st,const float* wt){
    int ot=(time+st-1)/st,of=(freq+sf-1)/sf;
    for(int t=0;t<ot;++t)for(int f=0;f<of;++f){float* out=y+(t*of+f)*channels;
        for(int kf=0;kf<3;++kf){int fi=f*sf+kf-1;if(fi<0||fi>=freq)continue;
            for(int kt=0;kt<3;++kt){int ti=t*st+kt-1;if(ti<0||ti>=time)continue;const float* in=x+(ti*freq+fi)*channels;const float* w=wt+(kf*3+kt)*channels;
                for(int c=0;c<channels;++c)out[c]+=in[c]*w[c];}}
    }
}
__attribute__((target("avx512f")))
void dw512(const float* x,float* y,int time,int freq,int channels,int sf,int st,const float* wt){
    int ot=(time+st-1)/st,of=(freq+sf-1)/sf;
    for(int t=0;t<ot;++t)for(int f=0;f<of;++f)for(int c=0;c<channels;c+=16){
        __mmask16 mask=(__mmask16)((1u<<std::min(16,channels-c))-1);__m512 sum=_mm512_setzero_ps();
        for(int kf=0;kf<3;++kf){int fi=f*sf+kf-1;if(fi<0||fi>=freq)continue;
            for(int kt=0;kt<3;++kt){int ti=t*st+kt-1;if(ti<0||ti>=time)continue;
                sum=_mm512_add_ps(sum,_mm512_mul_ps(_mm512_maskz_loadu_ps(mask,x+(ti*freq+fi)*channels+c),_mm512_maskz_loadu_ps(mask,wt+(kf*3+kt)*channels+c)));}}
        _mm512_mask_storeu_ps(y+(t*of+f)*channels+c,mask,sum);
    }
}
__attribute__((target("avx2")))
void dw256(const float* x,float* y,int time,int freq,int channels,int sf,int st,const float* wt){
    int ot=(time+st-1)/st,of=(freq+sf-1)/sf;
    for(int t=0;t<ot;++t)for(int f=0;f<of;++f){int c=0;for(;c+8<=channels;c+=8){__m256 sum=_mm256_setzero_ps();
        for(int kf=0;kf<3;++kf){int fi=f*sf+kf-1;if(fi<0||fi>=freq)continue;
            for(int kt=0;kt<3;++kt){int ti=t*st+kt-1;if(ti<0||ti>=time)continue;
                sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_loadu_ps(x+(ti*freq+fi)*channels+c),_mm256_loadu_ps(wt+(kf*3+kt)*channels+c)));}}
        _mm256_storeu_ps(y+(t*of+f)*channels+c,sum);}
        for(;c<channels;++c){float sum=0;for(int kf=0;kf<3;++kf){int fi=f*sf+kf-1;if(fi<0||fi>=freq)continue;for(int kt=0;kt<3;++kt){int ti=t*st+kt-1;if(ti<0||ti>=time)continue;sum+=x[(ti*freq+fi)*channels+c]*wt[(kf*3+kt)*channels+c];}}y[(t*of+f)*channels+c]=sum;}}
}
struct FrontendDispatch {BNKernel bn;DWKernel dw;const char* name;std::uint64_t key;};
FrontendDispatch select_frontend(){
    const char* forced=std::getenv("SHIELD_FRONTEND_SIMD");if(!forced)forced=std::getenv("SHIELD_KERNEL");
    __builtin_cpu_init();
    if(forced&&(std::strcmp(forced,"scalar")==0||std::strcmp(forced,"compiler")==0||std::strcmp(forced,"0")==0))return {batchnorm_activate,dw_compiler,"compiler",0};
    if((!forced||std::strcmp(forced,"avx2")!=0)&&__builtin_cpu_supports("avx512f"))return {bn512,dw512,"avx512",2};
    if(__builtin_cpu_supports("avx2"))return {bn256,dw256,"avx2",1};
    return {batchnorm_activate,dw_compiler,"compiler",0};
}
const FrontendDispatch frontend_dispatch=select_frontend();
std::uint64_t frontend_mode_key(){return execution_mode_key()|(frontend_dispatch.key<<32);}

constexpr int NT=997,NF=80;
struct ImagePool {
    std::array<std::vector<float>,4> free;
    std::vector<float> acquire(std::size_t count){
        int best=-1;for(int i=0;i<4;++i)if(free[i].size()>=count&&(best<0||free[i].size()<free[best].size()))best=i;
        if(best<0)for(int i=0;i<4;++i)if(best<0||free[i].size()>free[best].size())best=i;
        std::vector<float> result;result.swap(free[best]);if(result.size()<count)result.resize(count);return result;
    }
    void recycle(std::vector<float>& value){
        if(value.empty())return;
        int best=0;for(int i=1;i<4;++i)if(free[i].size()<free[best].size())best=i;
        if(free[best].empty()||free[best].size()<value.size())free[best].swap(value);
    }
    std::size_t bytes()const{std::size_t n=0;for(const auto& v:free)n+=v.capacity()*sizeof(float);return n;}
};
thread_local ImagePool image_pool;
struct FrontendScratch {std::vector<float> patches,mel,logits,pooled;};
thread_local FrontendScratch scratch;
struct Image {
    int t=0,f=0,c=0;
    std::vector<float> v;
    Image()=default;
    Image(int tt,int ff,int cc,bool zero=false):t(tt),f(ff),c(cc),v(image_pool.acquire(std::size_t(tt)*ff*cc)){
        if(zero)std::fill_n(v.data(),elements(),0.f);
    }
    Image(const Image&)=delete;
    Image& operator=(const Image&)=delete;
    Image(Image&& other)noexcept:t(other.t),f(other.f),c(other.c),v(std::move(other.v)){other.t=other.f=other.c=0;}
    Image& operator=(Image&& other)noexcept{
        if(this!=&other){image_pool.recycle(v);t=other.t;f=other.f;c=other.c;v=std::move(other.v);other.t=other.f=other.c=0;}return *this;
    }
    ~Image(){image_pool.recycle(v);}
    std::size_t elements()const{return std::size_t(t)*f*c;}
    float* at(int tt,int ff){return v.data()+(tt*f+ff)*c;}
    const float* at(int tt,int ff)const{return v.data()+(tt*f+ff)*c;}
};
struct BN {std::vector<float> scale,shift;};
struct Prepared {
    std::array<float,400> window;
    std::array<int,512> rev;
    std::array<float,256> wr,wi;
    std::array<std::vector<std::pair<int,float>>,80> filters;
    std::unordered_map<std::string,BN> norms;
    std::unordered_map<std::string,std::vector<float>> dw_weights;
    const Weights* weights;
    explicit Prepared(const Weights& w):weights(&w) {
        auto expect=[&](const std::string& name,std::initializer_list<int> shape){
            const auto& t=w.at(name);if(t.shape!=std::vector<int>(shape))throw std::runtime_error("invalid frontend weight shape: "+name);
            std::size_t count=1;for(int d:shape)count*=d;if(t.data.size()!=count)throw std::runtime_error("invalid frontend weight size: "+name);
        };
        auto bn=[&](const std::string& name,int c){for(const char* suffix:{"weight","bias","running_mean","running_var"})expect(name+"."+suffix,{c});};
        expect("frontend.mean",{80});expect("frontend.std",{80});
        if(w.tensors.count("frontend.window"))expect("frontend.window",{400});
        if(w.tensors.count("frontend.fb"))expect("frontend.fb",{80,257});
        expect("cnn.stem.0.weight",{32,1,3,3});bn("cnn.stem.1",32);
        int ci=32;const int channels[4]={48,96,160,192};
        for(int stage=0;stage<4;++stage)for(int block=0;block<2;++block){
            int co=channels[stage],hidden=ci*2;std::string base="cnn.stages."+std::to_string(stage)+"."+std::to_string(block);
            expect(base+".pw1.0.weight",{hidden,ci,1,1});bn(base+".pw1.1",hidden);
            expect(base+".dw.0.weight",{hidden,1,3,3});bn(base+".dw.1",hidden);
            auto& dw=dw_weights[base+".dw"];dw.resize(9*hidden);const auto& original=w.at(base+".dw.0.weight").data;
            for(int k=0;k<9;++k)for(int c=0;c<hidden;++c)dw[k*hidden+c]=original[c*9+k];
            expect(base+".pw2.0.weight",{co,hidden,1,1});bn(base+".pw2.1",co);ci=co;
        }
        expect("cnn.freq_attn.weight",{1,192,1,1});expect("cnn.freq_attn.bias",{1});expect("cnn.proj.weight",{384,192});
        auto win=w.tensors.find("frontend.window");
        for(int j=0;j<400;++j)window[j]=win==w.tensors.end()?0.5f*(1.f-std::cos(float(2.*M_PI*j/400.))):win->second.data[j];
        for(int j=0;j<512;++j){int q=j,r=0;for(int n=0;n<9;++n){r=r*2+(q&1);q>>=1;}rev[j]=r;}
        for(int j=0;j<256;++j){wr[j]=float(std::cos(-2.*M_PI*j/512.));wi[j]=float(std::sin(-2.*M_PI*j/512.));}
        auto fb=w.tensors.find("frontend.fb");
        std::array<double,82> hz;
        double m0=2595.*std::log10(1.+20./700.),m1=2595.*std::log10(1.+7600./700.);
        for(int j=0;j<82;++j)hz[j]=700.*(std::pow(10.,(m0+(m1-m0)*j/81.)/2595.)-1.);
        for(int m=0;m<80;++m)for(int k=0;k<257;++k){
            float v=fb!=w.tensors.end()?fb->second.data[m*257+k]:float(std::max(0.,std::min((k*31.25-hz[m])/(hz[m+1]-hz[m]),(hz[m+2]-k*31.25)/(hz[m+2]-hz[m+1]))));
            if(v!=0)filters[m].emplace_back(k,v);
        }
        for(auto& item:w.tensors){const auto& name=item.first;if(name.rfind("cnn.",0)!=0)continue;const std::string suffix=".running_mean";
            if(name.size()<suffix.size()||name.compare(name.size()-suffix.size(),suffix.size(),suffix)!=0)continue;
            std::string base=name.substr(0,name.size()-suffix.size());BN b;std::size_t n=item.second.data.size();b.scale.resize(n);b.shift.resize(n);
            auto& var=w.at(base+".running_var").data;auto& gamma=w.at(base+".weight").data;auto& beta=w.at(base+".bias").data;
            if(var.size()!=n||gamma.size()!=n||beta.size()!=n)throw std::runtime_error("inconsistent BatchNorm shape: "+base);
            for(std::size_t j=0;j<n;++j){b.scale[j]=gamma[j]/std::sqrt(var[j]+1e-5f);b.shift[j]=beta[j]-item.second.data[j]*b.scale[j];}
            norms.emplace(base,std::move(b));
        }
    }
};
std::unordered_map<const Weights*,std::shared_ptr<Prepared>> prep_cache;
std::mutex prep_mutex;
std::shared_ptr<Prepared> prepare(const Weights& w){std::lock_guard<std::mutex> l(prep_mutex);auto& p=prep_cache[&w];if(!p)p=std::make_shared<Prepared>(w);return p;}
inline float silu(float x){return x/(1.f+std::exp(-x));}
void batchnorm(Image& h,const Prepared& p,const std::string& name,bool activation){
    const auto& b=p.norms.at(name);int rows=h.t*h.f;
    frontend_dispatch.bn(h.v.data(),rows,h.c,b.scale.data(),b.shift.data(),activation);
}
Image pointwise(const Image& x,int out,const Prepared& p,const std::string& name,bool activation){
    Image y(x.t,x.f,out);const auto& w=*p.weights;
    linear(x.v.data(),y.v.data(),x.t*x.f,x.c,out,w.at(name+".0.weight").data.data());
    batchnorm(y,p,name+".1",activation);return y;
}
Image depthwise(const Image& x,int sf,int st,const Prepared& p,const std::string& name){
    Image y((x.t+st-1)/st,(x.f+sf-1)/sf,x.c,true);
    frontend_dispatch.dw(x.v.data(),y.v.data(),x.t,x.f,x.c,sf,st,p.dw_weights.at(name).data());
    batchnorm(y,p,name+".1",true);return y;
}
Image block(const Image& x,int out,int sf,int st,const Prepared& p,const std::string& name){
    auto stamp=std::chrono::steady_clock::now();
    auto report=[&](const char* op){if(std::getenv("SHIELD_FRONTEND_PROFILE"))std::fprintf(stderr,"%s %s %d %.3f ms\n",name.c_str(),op,x.t,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-stamp).count());stamp=std::chrono::steady_clock::now();};
    Image expanded=pointwise(x,x.c*2,p,name+".pw1",true);report("pw1");
    Image dw=depthwise(expanded,sf,st,p,name+".dw");report("dw");
    expanded=Image();
    Image result=pointwise(dw,out,p,name+".pw2",false);report("pw2");
    if(x.c==out&&sf==1&&st==1)for(std::size_t j=0;j<result.elements();++j)result.v[j]+=x.v[j];
    return result;
}
Image cnn_image(const std::vector<float>& mel,int begin,int end,const Prepared& p){
    int time=end-begin;
    // Conv2d's native order is output, input, frequency-kernel, time-kernel.
    auto& patches=scratch.patches;if(patches.size()<std::size_t(time)*80*9)patches.resize(std::size_t(time)*80*9);
    for(int t=0;t<time;++t)for(int f=0;f<80;++f)for(int kf=0;kf<3;++kf)for(int kt=0;kt<3;++kt){
        int ti=t+kt-1,fi=f+kf-1;
        patches[(t*80+f)*9+kf*3+kt]=(ti>=0&&ti<time&&fi>=0&&fi<80)?mel[(ti+begin)*80+fi]:0.f;
    }
    auto phase_start=std::chrono::steady_clock::now();
    Image h(time,80,32);
    linear(patches.data(),h.v.data(),time*80,9,32,p.weights->at("cnn.stem.0.weight").data.data());
    batchnorm(h,p,"cnn.stem.1",true);
    if(std::getenv("SHIELD_FRONTEND_PROFILE"))std::fprintf(stderr,"stem %d %.3f ms\n",time,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-phase_start).count());
    const int channels[4]={48,96,160,192};
    for(int s=0;s<4;++s){std::string base="cnn.stages."+std::to_string(s);
        phase_start=std::chrono::steady_clock::now();
        h=block(h,channels[s],s<2?2:1,s<3?2:1,p,base+".0");
        h=block(h,channels[s],1,1,p,base+".1");
        if(std::getenv("SHIELD_FRONTEND_PROFILE"))std::fprintf(stderr,"stage %d %d %.3f ms\n",s,time,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-phase_start).count());}
    return h;
}
std::vector<float> cnn(const std::vector<float>& mel,int begin,int end,const Prepared& p){
    Image h=cnn_image(mel,begin,end,p);int time=h.t;
    auto& logits=scratch.logits;auto& pooled=scratch.pooled;
    if(logits.size()<std::size_t(time)*20)logits.resize(std::size_t(time)*20);
    if(pooled.size()<std::size_t(time)*192)pooled.resize(std::size_t(time)*192);
    std::fill_n(pooled.data(),std::size_t(time)*192,0.f);std::vector<float> out(std::size_t(time)*384);
    linear(h.v.data(),logits.data(),time*20,192,1,p.weights->at("cnn.freq_attn.weight").data.data(),p.weights->at("cnn.freq_attn.bias").data.data());
    for(int t=0;t<time;++t){float max=-INFINITY;for(int f=0;f<20;++f)max=std::max(max,logits[t*20+f]);float sum=0;
        for(int f=0;f<20;++f){logits[t*20+f]=std::exp(logits[t*20+f]-max);sum+=logits[t*20+f];}
        for(int f=0;f<20;++f){float a=logits[t*20+f]/sum;const float* v=h.at(t,f);for(int c=0;c<192;++c)pooled[t*192+c]+=v[c]*a;}}
    linear(pooled.data(),out.data(),time,192,384,p.weights->at("cnn.proj.weight").data.data());return out;
}
std::vector<float> cnn_full(const std::vector<float>& mel,const Prepared& p){
    const char* env=std::getenv("SHIELD_CNN_TILE");
    int tile=env?std::atoi(env):25;
    if(tile==0)return cnn(mel,0,997,p);
    if(tile<1||tile>125)tile=25;
    std::vector<float> out(125*384);
    for(int a=0;a<125;a+=tile){int b=std::min(125,a+tile);
        int begin=std::max(0,a*8-40),end=std::min(997,(b-1)*8+40);
        auto slab=cnn(mel,begin,end,p);int offset=a-begin/8;
        std::copy(slab.begin()+offset*384,slab.begin()+(offset+b-a)*384,out.begin()+a*384);
    }
    return out;
}
void mel_frames(const float* pcm,std::vector<float>& mel,int first,const Prepared& p){
    const auto& mean=p.weights->at("frontend.mean").data;const auto& std=p.weights->at("frontend.std").data;
    for(int t=first;t<997;++t){std::array<float,512> real{},imag{};std::array<float,257> power{};
        for(int j=0;j<400;++j)real[p.rev[j+56]]=pcm[t*160+j+56]*p.window[j];
        for(int size=2;size<=512;size*=2){int half=size/2,step=512/size;
            for(int start=0;start<512;start+=size)for(int j=0;j<half;++j){int a=start+j,b=a+half,k=j*step;
                float r=p.wr[k]*real[b]-p.wi[k]*imag[b],i=p.wr[k]*imag[b]+p.wi[k]*real[b];
                float ra=real[a],ia=imag[a];real[a]=ra+r;imag[a]=ia+i;real[b]=ra-r;imag[b]=ia-i;}}
        for(int k=0;k<257;++k)power[k]=real[k]*real[k]+imag[k]*imag[k];
        for(int m=0;m<80;++m){float value=0;for(auto [k,w]:p.filters[m])value=std::fma(w,power[k],value);
            mel[t*80+m]=(std::log(value+1e-6f)-mean[m])/std[m];}
    }
}
}
const char* frontend_kernel_name(){return frontend_dispatch.name;}
std::size_t frontend_scratch_bytes(){return image_pool.bytes()+(scratch.patches.capacity()+scratch.mel.capacity()+scratch.logits.capacity()+scratch.pooled.capacity())*sizeof(float);}
void validate_frontend_weights(const Weights& w){(void)prepare(w);}
void release_frontend_weights(const Weights& w){std::lock_guard<std::mutex> lock(prep_mutex);prep_cache.erase(&w);}
void frontend_cnn(const Weights& w,const float* pcm,std::vector<float>& tokens,FrontendCache* cache){
    auto prep=prepare(w);bool reuse=cache&&cache->valid&&cache->weights==&w&&cache->execution_mode==frontend_mode_key()&&cache->tokens.size()==125*384&&cache->mel.size()==997*80&&
        (cache->trusted_shift||(cache->pcm.size()==160000&&std::memcmp(pcm,cache->pcm.data()+32000,128000*sizeof(float))==0));
    if(cache)cache->valid=false;
    auto& mel=cache?cache->mel:scratch.mel;mel.resize(997*80);
    if(reuse)std::memmove(mel.data(),mel.data()+200*80,797*80*sizeof(float));
    mel_frames(pcm,mel,reuse?797:0,*prep);
    if(reuse){tokens.resize(125*384);std::copy(cache->tokens.begin()+30*384,cache->tokens.begin()+120*384,tokens.begin()+5*384);
        // RF=77, temporal stride=8. Current [5,95) is unaffected by either
        // window boundary in the prior or current window. Slab starts are
        // multiples of 8, preserving the phase of every strided convolution.
        auto left=cnn(mel,0,72,*prep);auto right=cnn(mel,720,997,*prep);
        std::copy(left.begin(),left.begin()+5*384,tokens.begin());
        std::copy(right.begin()+5*384,right.end(),tokens.begin()+95*384);
    }else tokens=cnn_full(mel,*prep);
    if(const char* dir=std::getenv("SHIELD_DUMP_DIR")){
        std::vector<float> nchw(mel.size());for(int t=0;t<997;++t)for(int m=0;m<80;++m)nchw[m*997+t]=mel[t*80+m];
        std::ofstream(std::string(dir)+"/mel.f32",std::ios::binary).write(reinterpret_cast<const char*>(nchw.data()),nchw.size()*sizeof(float));
        std::ofstream(std::string(dir)+"/cnn.f32",std::ios::binary).write(reinterpret_cast<const char*>(tokens.data()),tokens.size()*sizeof(float));
    }
    if(cache){if(!cache->trusted_shift)cache->pcm.assign(pcm,pcm+160000);cache->tokens=tokens;cache->weights=&w;cache->execution_mode=frontend_mode_key();cache->valid=true;cache->reused_tokens=reuse?90:0;cache->computed_mel_frames=reuse?200:997;}
}
}
