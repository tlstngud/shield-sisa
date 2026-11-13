#include "core.hpp"
#include "sequence_cache.hpp"
#include "precision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <vector>

namespace shield {
namespace {
constexpr int T=126, D=384, H=6, K=64, V=128, VD=H*V, FF=960, S=32;
using Vec=std::vector<float>;
inline float silu(float x) { return x/(1.0f+std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f/(1.0f+std::exp(-x)); }
inline float softplus(float x) { return x>20.0f ? x : std::log1p(std::exp(x)); }
const float* data(const Weights& w,const std::string& n) { return w.at(n).data.data(); }
void project(const Weights& w,const std::string& n,const float* x,float* y,int rows,int in,int out,bool bias=false) {
    linear(x,y,rows,in,out,data(w,n+".weight"),bias?data(w,n+".bias"):nullptr);
}
void norm(const float* x,float* y,int rows,int dim,const float* weight,float eps=1e-6f) {
    for(int r=0;r<rows;++r) {
        const float* a=x+r*dim; float* b=y+r*dim;
        float sum=0; for(int j=0;j<dim;++j) sum+=a[j]*a[j];
        const float inv=1.0f/std::sqrt(sum/dim+eps);
        for(int j=0;j<dim;++j) b[j]=(a[j]*inv)*weight[j];
    }
}
void dump(const char* name,const float* p,size_t n) {
    const char* dir=std::getenv("SHIELD_DUMP_DIR");
    if(!dir || !*dir) return;
    std::ofstream f(std::string(dir)+"/"+name+".f32",std::ios::binary);
    f.write(reinterpret_cast<const char*>(p),n*sizeof(float));
}
struct Workspace {
    Vec h=Vec(T*D), normalized=Vec(T*D), branch=Vec(T*D);
    Vec q=Vec(T*D), k=Vec(T*D), v=Vec(T*VD), gate=Vec(T*VD), o=Vec(T*VD);
    Vec a=Vec(T*H), b=Vec(T*H), conv=Vec(T*VD), state=Vec(H*K*V);
    Vec fgate=Vec(T*FF), fup=Vec(T*FF);
    Vec sk=Vec(T*D), sv=Vec(T*D), sb=Vec(T*H*S), theta=Vec(T*H*S/2);
    Vec decay=Vec(T*H), forget=Vec(T*H), phi=Vec(T*H*S/2);
    Vec last=Vec(D), sq=Vec(D), sc=Vec(H*S), attended=Vec(D), scores=Vec(T);
    Vec hidden=Vec(128);
};
// Workspace belongs to the calling worker, never a stream; every stateful quantity
// is overwritten/reset for each window and each repeated mixer invocation.
thread_local Workspace ws;

void delta_scalar(float* state,const float* k,const float* q,const float* v,float a,float beta,float* out) {
    float u[V]={}; std::fill_n(out,V,0.0f);
    for(int j=0;j<K;++j) {
        float* st=state+j*V; const float kj=k[j];
        for(int l=0;l<V;++l) { st[l]*=a; u[l]+=st[l]*kj; }
    }
    for(int l=0;l<V;++l) u[l]=beta*(v[l]-u[l]);
    for(int j=0;j<K;++j) {
        float* st=state+j*V; const float kj=k[j], qj=q[j];
        for(int l=0;l<V;++l) { st[l]+=kj*u[l]; out[l]+=st[l]*qj; }
    }
}
__attribute__((target("avx512f,fma")))
void delta_avx512(float* state,const float* k,const float* q,const float* v,float a,float beta,float* out) {
    __m512 u[8], o[8]; const __m512 decay=_mm512_set1_ps(a), bt=_mm512_set1_ps(beta);
    for(int z=0;z<8;++z) u[z]=o[z]=_mm512_setzero_ps();
    for(int j=0;j<K;++j) {
        const __m512 kj=_mm512_set1_ps(k[j]);
        for(int z=0;z<8;++z) {
            float* p=state+j*V+z*16;
            const __m512 st=_mm512_mul_ps(_mm512_loadu_ps(p),decay);
            _mm512_storeu_ps(p,st); u[z]=_mm512_fmadd_ps(st,kj,u[z]);
        }
    }
    for(int z=0;z<8;++z) u[z]=_mm512_mul_ps(bt,_mm512_sub_ps(_mm512_loadu_ps(v+z*16),u[z]));
    for(int j=0;j<K;++j) {
        const __m512 kj=_mm512_set1_ps(k[j]), qj=_mm512_set1_ps(q[j]);
        for(int z=0;z<8;++z) {
            float* p=state+j*V+z*16;
            const __m512 st=_mm512_fmadd_ps(kj,u[z],_mm512_loadu_ps(p));
            _mm512_storeu_ps(p,st); o[z]=_mm512_fmadd_ps(st,qj,o[z]);
        }
    }
    for(int z=0;z<8;++z) _mm512_storeu_ps(out+z*16,o[z]);
}
__attribute__((target("avx2,fma")))
void delta_avx2(float* state,const float* k,const float* q,const float* v,float a,float beta,float* out) {
    const __m256 decay=_mm256_set1_ps(a), bt=_mm256_set1_ps(beta);
    for(int base=0;base<V;base+=32) {
        __m256 u[4], o[4]; for(int z=0;z<4;++z) u[z]=o[z]=_mm256_setzero_ps();
        for(int j=0;j<K;++j) {
            const __m256 kj=_mm256_set1_ps(k[j]);
            for(int z=0;z<4;++z) {
                float* p=state+j*V+base+z*8;
                const __m256 st=_mm256_mul_ps(_mm256_loadu_ps(p),decay);
                _mm256_storeu_ps(p,st); u[z]=_mm256_fmadd_ps(st,kj,u[z]);
            }
        }
        for(int z=0;z<4;++z) u[z]=_mm256_mul_ps(bt,_mm256_sub_ps(_mm256_loadu_ps(v+base+z*8),u[z]));
        for(int j=0;j<K;++j) {
            const __m256 kj=_mm256_set1_ps(k[j]), qj=_mm256_set1_ps(q[j]);
            for(int z=0;z<4;++z) {
                float* p=state+j*V+base+z*8;
                const __m256 st=_mm256_fmadd_ps(kj,u[z],_mm256_loadu_ps(p));
                _mm256_storeu_ps(p,st); o[z]=_mm256_fmadd_ps(st,qj,o[z]);
            }
        }
        for(int z=0;z<4;++z) _mm256_storeu_ps(out+base+z*8,o[z]);
    }
}
using DeltaHead=void(*)(float*,const float*,const float*,const float*,float,float,float*);
DeltaHead delta_kernel() {
    const char* forced=std::getenv("SHIELD_DELTA_KERNEL");
    if(!forced) forced=std::getenv("SHIELD_KERNEL");
    if(forced && std::string(forced)=="scalar") return delta_scalar;
    if((!forced || std::string(forced)!="avx2") && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) return delta_avx512;
    if(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return delta_avx2;
    return delta_scalar;
}

void short_conv(const Weights& w,const std::string& n,Vec& x,int channels,Vec& tmp,
                const Vec* saved=nullptr) {
    const float* weight=data(w,n+".weight"); const float* bias=data(w,n+".bias");
    for(int t=0;t<T;++t) {
        if(saved&&t>=8&&t<95){std::copy_n(saved->data()+(t-8)*channels,channels,tmp.data()+t*channels);continue;}
        for(int c=0;c<channels;++c) {
            float val=bias[c];
            for(int j=0;j<4;++j) if(t+j>=3) val+=x[(t+j-3)*channels+c]*weight[c*4+j];
            tmp[t*channels+c]=silu(val);
        }
    }
    std::copy_n(tmp.data(),T*channels,x.data());
}
void l2_norm(Vec& x,bool skip_cached=false) {
    for(int t=0;t<T;++t) {
        if(skip_cached&&t>=8&&t<95)continue;
        for(int h=0;h<H;++h){float* p=x.data()+(t*H+h)*K;float sum=0;
            for(int j=0;j<K;++j)sum+=p[j]*p[j];
            float inv=1.0f/std::max(std::sqrt(sum),1e-12f);
            for(int j=0;j<K;++j)p[j]*=inv;}
    }
}
constexpr int PROJ_WIDTHS[6]={D,D,VD,H,H,VD};
const char* PROJ_NAMES[6]={"q_proj","k_proj","v_proj","a_proj","b_proj","g_proj"};
std::array<Vec*,6> projected_workspace(){return {&ws.q,&ws.k,&ws.v,&ws.a,&ws.b,&ws.gate};}
void transform_gates(const Weights& w,const std::string& n,float* a,float* b,float* gate,int rows){
    const float* A=data(w,n+".A_log");const float* dt=data(w,n+".dt_bias");
    float rates[H];for(int h=0;h<H;++h)rates[h]=std::exp(A[h]);
    for(int t=0;t<rows;++t){
        for(int h=0;h<H;++h){a[t*H+h]=std::exp(-rates[h]*softplus(a[t*H+h]+dt[h]));b[t*H+h]=sigmoid(b[t*H+h]);}
        for(int j=0;j<VD;++j)gate[t*VD+j]=silu(gate[t*VD+j]);
    }
}
struct SharedReadout {std::array<Vec,6> projected;};
std::mutex readout_mutex;
std::unordered_map<const Weights*,std::unordered_map<std::uint64_t,std::shared_ptr<SharedReadout>>> readouts;
std::shared_ptr<SharedReadout> prepared_readout(const Weights& w){
    std::lock_guard<std::mutex> lock(readout_mutex);auto& ptr=readouts[&w][execution_mode_key()];
    if(!ptr){auto created=std::make_shared<SharedReadout>();float normalized[D];
        norm(data(w,"readout"),normalized,1,D,data(w,"gdn_a.mixer_norm.weight"));
        for(int i=0;i<6;++i){created->projected[i].resize(PROJ_WIDTHS[i]);
            project(w,std::string("gdn_a.mixer.")+PROJ_NAMES[i],normalized,created->projected[i].data(),1,D,PROJ_WIDTHS[i]);}
        transform_gates(w,"gdn_a.mixer",created->projected[3].data(),created->projected[4].data(),created->projected[5].data(),1);
        ptr=std::move(created);
    }return ptr;
}
void mixer(const Weights& w,const std::string& n,const float* x,float* output,SequenceCache* cache=nullptr,bool reuse=false) {
    const auto projected=projected_workspace();
    const bool caching=cache&&cache->level>0;
    std::shared_ptr<SharedReadout> readout;if(caching)readout=prepared_readout(w);
    for(int i=0;i<6;++i){float* dest=projected[i]->data();int width=PROJ_WIDTHS[i];
        if(reuse){
            project(w,n+"."+PROJ_NAMES[i],x,dest,5,D,width);
            std::copy_n(cache->projected[i].data(),90*width,dest+5*width);
            project(w,n+"."+PROJ_NAMES[i],x+95*D,dest+95*width,30,D,width);
        }else project(w,n+"."+PROJ_NAMES[i],x,dest,caching?125:T,D,width);
        if(caching)std::copy_n(readout->projected[i].data(),width,dest+125*width);
    }
    if(reuse){
        transform_gates(w,n,ws.a.data(),ws.b.data(),ws.gate.data(),5);
        transform_gates(w,n,ws.a.data()+95*H,ws.b.data()+95*H,ws.gate.data()+95*VD,30);
    }else transform_gates(w,n,ws.a.data(),ws.b.data(),ws.gate.data(),caching?125:T);
    if(caching)for(int i=0;i<6;++i){int width=PROJ_WIDTHS[i];cache->projected[i].resize(90*width);
        std::copy_n(projected[i]->data()+30*width,90*width,cache->projected[i].data());}
    const bool reuse_processed=reuse&&cache->level==2;
    short_conv(w,n+".q_conv1d",ws.q,D,ws.conv,reuse_processed?&cache->processed[0]:nullptr);
    short_conv(w,n+".k_conv1d",ws.k,D,ws.conv,reuse_processed?&cache->processed[1]:nullptr);
    short_conv(w,n+".v_conv1d",ws.v,VD,ws.conv,reuse_processed?&cache->processed[2]:nullptr);
    l2_norm(ws.q,reuse_processed);l2_norm(ws.k,reuse_processed);
    if(caching&&cache->level==2)for(int i=0;i<3;++i){int width=PROJ_WIDTHS[i];cache->processed[i].resize(87*width);
        std::copy_n(projected[i]->data()+33*width,87*width,cache->processed[i].data());}
    const float* ow=data(w,n+".o_norm.weight");
    const DeltaHead delta=delta_kernel();
    std::fill(ws.state.begin(),ws.state.end(),0.0f);
    for(int t=0;t<T;++t) for(int h=0;h<H;++h) {
        const float a=ws.a[t*H+h],beta=ws.b[t*H+h];
        float* state=ws.state.data()+h*K*V;
        const float* k=ws.k.data()+(t*H+h)*K;
        const float* q=ws.q.data()+(t*H+h)*K;
        const float* v=ws.v.data()+(t*H+h)*V;
        float out[V]; delta(state,k,q,v,a,beta,out);
        float sum=0; for(int l=0;l<V;++l) { out[l]*=0.125f; sum+=out[l]*out[l]; }
        const float inv=1.0f/std::sqrt(sum/V+1e-5f);
        float* o=ws.o.data()+(t*H+h)*V;
        const float* gate=ws.gate.data()+(t*H+h)*V;
        for(int l=0;l<V;++l) o[l]=((out[l]*inv)*ow[l])*gate[l];
    }
    project(w,n+".o_proj",ws.o.data(),output,T,VD,D);
}
void ffn(const Weights& w,const std::string& n,const float* x,float* y,int rows) {
    project(w,n+".w_gate",x,ws.fgate.data(),rows,D,FF);
    project(w,n+".w_up",x,ws.fup.data(),rows,D,FF);
    for(int i=0;i<rows*FF;++i) ws.fgate[i]=silu(ws.fgate[i])*ws.fup[i];
    project(w,n+".w_down",ws.fgate.data(),y,rows,FF,D);
}
void block(const Weights& w,const std::string& n,int repeats,float scale,SequenceCache* cache=nullptr,bool reuse=false) {
    for(int r=0;r<repeats;++r) {
        norm(ws.h.data(),ws.normalized.data(),T,D,data(w,n+".mixer_norm.weight"));
        mixer(w,n+".mixer",ws.normalized.data(),ws.branch.data(),cache,reuse);
        for(int i=0;i<T*D;++i) ws.h[i]+=scale*ws.branch[i];
    }
    norm(ws.h.data(),ws.normalized.data(),T,D,data(w,n+".ffn_norm.weight"));
    ffn(w,n+".ffn",ws.normalized.data(),ws.branch.data(),T);
    for(int i=0;i<T*D;++i) ws.h[i]+=ws.branch[i];
}
void sisa_last(const Weights& w) {
    const std::string n="sisa.attn";
    norm(ws.h.data(),ws.normalized.data(),T,D,data(w,"sisa.attn_norm.weight"));
    const float* x=ws.normalized.data(); const float* last=x+(T-1)*D;
    project(w,n+".W_Q",last,ws.sq.data(),1,D,D);
    project(w,n+".W_C",last,ws.sc.data(),1,D,H*S);
    project(w,n+".W_K",x,ws.sk.data(),T,D,D);
    project(w,n+".W_V",x,ws.sv.data(),T,D,D);
    project(w,n+".W_B",x,ws.sb.data(),T,D,H*S);
    project(w,n+".theta_proj",x,ws.theta.data(),T,D,H*S/2);
    project(w,n+".alpha_proj",x,ws.decay.data(),T,D,H,true);
    project(w,n+".forget_proj",x,ws.forget.data(),T,D,H,true);
    // torch.cumsum computes float32 prefixes using a wider accumulator.
    for(int h=0;h<H;++h) {
        double g=0, f=0; double phi[S/2]={};
        for(int t=0;t<T;++t) {
            g-=softplus(ws.decay[t*H+h]); f-=softplus(-ws.forget[t*H+h]);
            ws.decay[t*H+h]=float(g); ws.forget[t*H+h]=float(f);
            for(int p=0;p<S/2;++p) {
                const int i=(t*H+h)*(S/2)+p; phi[p]+=ws.theta[i]; ws.phi[i]=float(phi[p]);
                const float co=std::cos(ws.phi[i]), si=std::sin(ws.phi[i]);
                float* b=ws.sb.data()+(t*H+h)*S+p*2;
                const float e=b[0], o=b[1]; b[0]=e*co-o*si; b[1]=e*si+o*co;
                if(t==T-1) {
                    float* c=ws.sc.data()+h*S+p*2; const float ce=c[0], codd=c[1];
                    c[0]=ce*co-codd*si; c[1]=ce*si+codd*co;
                }
            }
        }
    }
    const float* lambda=data(w,n+".lambda_raw");
    for(int h=0;h<H;++h) {
        const float* q=ws.sq.data()+h*K; const float* c=ws.sc.data()+h*S;
        const float lam=softplus(lambda[h]); float maximum=-INFINITY;
        for(int t=0;t<T;++t) {
            const float* k=ws.sk.data()+(t*H+h)*K; const float* b=ws.sb.data()+(t*H+h)*S;
            float content=0, ssm=0;
            for(int j=0;j<K;++j) content+=q[j]*k[j];
            for(int j=0;j<S;++j) ssm+=c[j]*b[j];
            const float score=content*0.125f + lam*std::exp(ws.decay[(T-1)*H+h]-ws.decay[t*H+h])*ssm
                             +(ws.forget[(T-1)*H+h]-ws.forget[t*H+h]);
            ws.scores[t]=score; maximum=std::max(maximum,score);
        }
        float sum=0; for(int t=0;t<T;++t) { ws.scores[t]=std::exp(ws.scores[t]-maximum); sum+=ws.scores[t]; }
        float* out=ws.attended.data()+h*K; std::fill_n(out,K,0.0f);
        for(int t=0;t<T;++t) {
            const float prob=ws.scores[t]/sum; const float* v=ws.sv.data()+(t*H+h)*K;
            for(int j=0;j<K;++j) out[j]+=prob*v[j];
        }
    }
    project(w,n+".W_O",ws.attended.data(),ws.branch.data(),1,D,D);
    for(int j=0;j<D;++j) ws.last[j]=ws.h[(T-1)*D+j]+ws.branch[j];
    norm(ws.last.data(),ws.normalized.data(),1,D,data(w,"sisa.ffn_norm.weight"));
    ffn(w,"sisa.ffn",ws.normalized.data(),ws.branch.data(),1);
    for(int j=0;j<D;++j) ws.last[j]+=ws.branch[j];
}
} // namespace

void validate_sequence_weights(const Weights& w) {
    auto check=[&](const std::string& name,std::initializer_list<int> expected) {
        const auto& tensor=w.at(name);
        if(tensor.shape!=std::vector<int>(expected)) throw std::runtime_error("invalid sequence weight shape: "+name);
    };
    auto lin=[&](const std::string& name,int in,int out,bool bias=false) {
        check(name+".weight",{out,in}); if(bias) check(name+".bias",{out});
    };
    check("readout",{1,1,D});
    for(const std::string block:{"gdn_a","gdn_b"}) {
        check(block+".mixer_norm.weight",{D}); check(block+".ffn_norm.weight",{D});
        const std::string mixer=block+".mixer";
        lin(mixer+".q_proj",D,D); lin(mixer+".k_proj",D,D); lin(mixer+".v_proj",D,VD);
        lin(mixer+".a_proj",D,H); lin(mixer+".b_proj",D,H); lin(mixer+".g_proj",D,VD); lin(mixer+".o_proj",VD,D);
        check(mixer+".A_log",{H}); check(mixer+".dt_bias",{H}); check(mixer+".o_norm.weight",{V});
        for(const std::string conv:{"q_conv1d","k_conv1d","v_conv1d"}) {
            const int width=conv=="v_conv1d"?VD:D;
            check(mixer+"."+conv+".weight",{width,1,4}); check(mixer+"."+conv+".bias",{width});
        }
        lin(block+".ffn.w_gate",D,FF); lin(block+".ffn.w_up",D,FF); lin(block+".ffn.w_down",FF,D);
    }
    check("sisa.attn_norm.weight",{D}); check("sisa.ffn_norm.weight",{D});
    for(const std::string proj:{"W_Q","W_K","W_V","W_O"}) lin("sisa.attn."+proj,D,D);
    lin("sisa.attn.W_B",D,H*S); lin("sisa.attn.W_C",D,H*S); lin("sisa.attn.theta_proj",D,H*S/2);
    lin("sisa.attn.alpha_proj",D,H,true); lin("sisa.attn.forget_proj",D,H,true); check("sisa.attn.lambda_raw",{H});
    lin("sisa.ffn.w_gate",D,FF); lin("sisa.ffn.w_up",D,FF); lin("sisa.ffn.w_down",FF,D);
    check("head_norm.weight",{D}); lin("head.0",D,128,true); lin("head.3",128,1,true);
}

static void run_sequence(const Weights& w,const float* tokens,float& logit,double* times,SequenceCache* cache=nullptr,bool reuse_frontend=false) {
    using Clock=std::chrono::steady_clock;
    if(cache&&(cache->level<0||cache->level>2))throw std::invalid_argument("sequence cache level must be 0, 1 or 2");
    std::copy_n(tokens,(T-1)*D,ws.h.data());
    std::copy_n(data(w,"readout"),D,ws.h.data()+(T-1)*D);
    dump("tokens",ws.h.data(),T*D);
    auto begin=times?Clock::now():Clock::time_point{};
    bool reuse=cache&&cache->level>0&&reuse_frontend&&cache->valid&&cache->weights==&w&&cache->execution_mode==execution_mode_key()&&cache->cached_level==cache->level;
    if(cache&&cache->level>0){for(int i=0;i<6;++i)reuse=reuse&&cache->projected[i].size()==std::size_t(90*PROJ_WIDTHS[i]);
        if(cache->level==2)for(int i=0;i<3;++i)reuse=reuse&&cache->processed[i].size()==std::size_t(87*PROJ_WIDTHS[i]);}
    if(cache){cache->valid=false;cache->reused_projection_rows=cache->reused_conv_rows=0;}
    block(w,"gdn_a",1,1.0f,cache,reuse);
    if(times) {const auto now=Clock::now(); times[0]=std::chrono::duration<double,std::milli>(now-begin).count(); begin=now;}
    dump("gdn_a",ws.h.data(),T*D);
    block(w,"gdn_b",2,0.7071067811865475244f);
    if(times) {const auto now=Clock::now(); times[1]=std::chrono::duration<double,std::milli>(now-begin).count(); begin=now;}
    dump("gdn_b",ws.h.data(),T*D);
    sisa_last(w);
    if(times) {const auto now=Clock::now(); times[2]=std::chrono::duration<double,std::milli>(now-begin).count(); begin=now;}
    dump("sisa_last",ws.last.data(),D);
    norm(ws.last.data(),ws.normalized.data(),1,D,data(w,"head_norm.weight"));
    project(w,"head.0",ws.normalized.data(),ws.hidden.data(),1,D,128,true);
    constexpr float inv_sqrt2=0.7071067811865475244f;
    for(float& z:ws.hidden) z=0.5f*z*(1.0f+std::erf(z*inv_sqrt2));
    project(w,"head.3",ws.hidden.data(),&logit,1,128,1,true);
    if(times) times[3]=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
    dump("logit",&logit,1);
    if(cache&&cache->level>0){cache->weights=&w;cache->execution_mode=execution_mode_key();cache->cached_level=cache->level;cache->valid=true;
        cache->reused_projection_rows=reuse?90:0;cache->reused_conv_rows=reuse&&cache->level==2?87:0;}
}
void sequence(const Weights& w,const float* tokens,float& logit) {run_sequence(w,tokens,logit,nullptr);}
void sequence_profile(const Weights& w,const float* tokens,float& logit,double stage_ms[4]) {
    run_sequence(w,tokens,logit,stage_ms);
}
void sequence_cached(const Weights& w,const float* tokens,float& logit,SequenceCache* cache,bool reuse_frontend){run_sequence(w,tokens,logit,nullptr,cache,reuse_frontend);}
void sequence_cached_profile(const Weights& w,const float* tokens,float& logit,SequenceCache* cache,bool reuse_frontend,double stage_ms[4]){run_sequence(w,tokens,logit,stage_ms,cache,reuse_frontend);}
void release_sequence_weights(const Weights& w){std::lock_guard<std::mutex> lock(readout_mutex);readouts.erase(&w);}
std::size_t sequence_shared_bytes(const Weights& w){std::lock_guard<std::mutex> lock(readout_mutex);std::size_t n=0;auto found=readouts.find(&w);if(found!=readouts.end())for(auto& mode:found->second)for(auto& p:mode.second->projected)n+=p.capacity()*sizeof(float);return n;}
} // namespace shield
