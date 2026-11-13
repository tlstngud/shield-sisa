#include "include/shield.h"
#include "core.hpp"
#include "frontend.hpp"
#include "precision.hpp"
#include "sequence_cache.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
thread_local std::string error;
constexpr float missing_float=std::numeric_limits<float>::quiet_NaN();
void check_pcm(const float* pcm,size_t n) {
    if (!pcm && n) throw std::invalid_argument("null audio");
    for (size_t i=0;i<n;++i) if (!std::isfinite(pcm[i]) || std::abs(pcm[i])>1.0f)
        throw std::invalid_argument("audio must be finite float32 in [-1,1]");
}
class Model {
public:
    shield::Weights weights;
    const shield::ComputeMode mode;
    explicit Model(const char* path,int workers,shield::ComputeMode precision,bool experimental):weights(path),mode(precision) {
        try {
            if(mode!=shield::ComputeMode::Exact&&!experimental){
                std::ifstream cert(std::string(path)+".cert");std::string magic,hash,name,revision;bool passed=false;
                if(cert>>magic && magic=="SHIELD_CERT_V1")while(cert>>hash>>name>>revision)
                    passed|=hash==weights.fingerprint && name==shield::compute_mode_name(mode) && revision==shield::compute_revision();
                if(!passed)throw std::runtime_error("mode has no passing quality certificate for these exported weights");
            }
            shield::validate_sequence_weights(weights);
            shield::validate_frontend_weights(weights);
            shield::register_compute_weights(weights);
            for (int i=0;i<workers;++i) workers_.emplace_back([this]{
                for (;;) {
                    std::function<void()> job;
                    { std::unique_lock<std::mutex> lock(mutex_);
                      cv_.wait(lock,[this]{return stop_ || !queue_.empty();});
                      if (stop_ && queue_.empty()) return;
                      job=std::move(queue_.front()); queue_.pop_front(); }
                    job();
                }
            });
        } catch (...) { stop();shield::release_sequence_weights(weights);shield::release_frontend_weights(weights);shield::release_kernel_weights(weights);throw; }
    }
    ~Model(){stop();shield::release_sequence_weights(weights);shield::release_frontend_weights(weights);shield::release_kernel_weights(weights);}
    float run(const float* pcm,shield::FrontendCache* cache=nullptr,size_t ring_start=0,shield::SequenceCache* seq_cache=nullptr) {
            shield::ScopedComputeMode precision(mode);
            thread_local std::vector<float> tokens;
            // Linearization is worker scratch, not another 640 KB per participant.
            thread_local std::vector<float> window;
            const float* input=pcm;
            if(ring_start){window.resize(160000);
                std::copy_n(pcm+ring_start,160000-ring_start,window.data());
                std::copy_n(pcm,ring_start,window.data()+160000-ring_start);input=window.data();}
            shield::frontend_cnn(weights,input,tokens,cache);
            float logit; shield::sequence_cached(weights,tokens.data(),logit,seq_cache,cache&&cache->reused_tokens==90);
            if (!std::isfinite(logit)) throw std::runtime_error("nonfinite model output");
            return logit;
    }
    void enqueue(std::function<void()> job){
        {std::lock_guard<std::mutex> lock(mutex_);queue_.emplace_back(std::move(job));}
        cv_.notify_one();
    }
    std::future<float> submit(const float* pcm,shield::FrontendCache* cache=nullptr,size_t ring_start=0,shield::SequenceCache* seq_cache=nullptr) {
        auto task=std::make_shared<std::packaged_task<float()>>([this,pcm,cache,ring_start,seq_cache]{return run(pcm,cache,ring_start,seq_cache);});
        auto result=task->get_future();
        enqueue([task]{(*task)();});return result;
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> workers_;
    bool stop_=false;
    void stop(){
        { std::lock_guard<std::mutex> lock(mutex_); stop_=true; }
        cv_.notify_all(); for(auto& t:workers_) if(t.joinable()) t.join();
    }
};
double speech(const float* pcm,size_t samples) {
    const size_t frames=samples/320;
    if (!frames) return 0;
    std::array<double,500> energies;
    if (frames>energies.size()) throw std::invalid_argument("VAD supports at most 10 seconds");
    for(size_t f=0;f<frames;++f) {
        double sum=0; for(size_t k=0;k<320;++k){double x=pcm[f*320+k];sum+=x*x;}
        energies[f]=10*std::log10(sum/320+1e-12);
    }
    auto ordered=energies;
    std::sort(ordered.begin(),ordered.begin()+frames);
    double index=(frames-1)*0.1;
    size_t lo=static_cast<size_t>(index),hi=std::min(lo+1,frames-1);
    const double floor=ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo);
    const double threshold=std::max(floor+12.0,-50.0);
    size_t active=0;for(size_t i=0;i<frames;++i)active+=energies[i]>threshold;
    return active*0.02;
}
// SHIELD's training windows pass a 7.2 kHz Butterworth low-pass (order 8, scipy sosfiltfilt). A stream cannot run a
// zero-phase filter without re-filtering every window (and losing the 2 s frontend reuse), so streams apply the same
// four sections forward twice: |H|^2 like sosfiltfilt, causal phase. Coefficients are scipy 1.15.3
// butter(8, 7200/8000, output="sos"); the recurrence is scipy sosfilt's transposed direct form II in double.
constexpr double lowpass_sos[4][6]={
    {0.44508427467139355,0.8901685493427871,0.44508427467139355,1.0,1.459706254376867,0.5348259849616106},
    {1.0,2.0,1.0,1.0,1.5132907658389043,0.5911680745682052},
    {1.0,2.0,1.0,1.0,1.623405697641,0.7069497656826819},
    {1.0,2.0,1.0,1.0,1.7939618452517716,0.8862831120070138}};
// Training stores the filtered windows as int16 clipped to +-32767, so the model never saw the filter's overshoot past
// full scale; the output (not the filter state) is clipped the same way.
constexpr double lowpass_peak=32767.0/32768.0;
void lowpass_run(std::array<double,16>& z,const float* in,float* out,size_t n){
    for(size_t i=0;i<n;++i){
        double v=in[i];
        for(int k=0;k<8;++k){
            const double* c=lowpass_sos[k%4];double* q=z.data()+2*k;
            const double y=c[0]*v+q[0];
            q[0]=c[1]*v-c[4]*y+q[1];
            q[1]=c[2]*v-c[5]*y;
            v=y;
        }
        out[i]=static_cast<float>(std::clamp(v,-lowpass_peak,lowpass_peak));
    }
}
}

struct shield_model {std::shared_ptr<Model> impl;};
struct shield_stream {
    std::shared_ptr<Model> model;
    std::string id,version;
    double threshold;
    int alert_k,alert_n;
    std::vector<float> ring=std::vector<float>(160000);
    // Optional low-passed copy of the ring that the model reads; VAD keeps reading the raw ring.
    bool lowpass=false;
    std::vector<float> lp_ring;
    std::array<double,16> lp_state{};
    const float* model_ring()const{return lowpass?lp_ring.data():ring.data();}
    size_t write=0,filled=0,since=0;
    std::vector<unsigned char> recent;
    shield::FrontendCache cache;
    shield::SequenceCache sequence_cache;
    mutable std::mutex mutex;
    std::condition_variable available;
    bool busy=false;
};
struct TicketTiming {
    std::chrono::steady_clock::time_point created=std::chrono::steady_clock::now(),started,finished;
};
struct shield_ticket {
    std::future<shield_result> future;
    std::shared_ptr<Model> owner;
    std::shared_ptr<TicketTiming> timing;
    shield_result result{};
    bool resolved=false;
};

extern "C" {
const char* shield_last_error(){return error.c_str();}
const char* shield_kernel(){return shield::kernel_name();}
const char* shield_compute_revision(){return shield::compute_revision();}
const char* shield_model_mode(const shield_model* m){return m?shield::compute_mode_name(m->impl->mode):nullptr;}
const char* shield_model_fingerprint(const shield_model* m){return m?m->impl->weights.fingerprint.c_str():nullptr;}
static shield_model* load_model(const char* path,int workers,const char* name,bool experimental){
    try { if(!path || workers<1 || workers>8)throw std::invalid_argument("path and 1..8 workers required");
        if(!name)throw std::invalid_argument("mode required");
        shield::ComputeMode mode=shield::ComputeMode::Exact;bool known=false;
        for(int i=0;i<=5;++i)if(std::string(name)==shield::compute_mode_name(static_cast<shield::ComputeMode>(i))){mode=static_cast<shield::ComputeMode>(i);known=true;}
        if(!known)throw std::invalid_argument("unknown compute mode");
        auto m=std::make_unique<shield_model>();m->impl=std::make_shared<Model>(path,workers,mode,experimental);error.clear();return m.release();
    }catch(const std::exception& e){error=e.what();return nullptr;}
}
shield_model* shield_model_load(const char* path,int workers){return load_model(path,workers,"exact",false);}
shield_model* shield_model_load_mode(const char* path,int workers,const char* mode){return load_model(path,workers,mode,false);}
shield_model* shield_model_load_experimental(const char* path,int workers,const char* mode){
#ifdef SHIELD_ENABLE_EXPERIMENTAL
    return load_model(path,workers,mode,true);
#else
    (void)path;(void)workers;(void)mode;error="uncertified modes are disabled in this build";return nullptr;
#endif
}
void shield_model_free(shield_model* m){delete m;}
size_t shield_weight_bytes(const shield_model* m){return m?m->impl->weights.bytes():0;}
int shield_predict(shield_model* m,const float* pcm,float* logit){
    if(logit)*logit=missing_float;
    try {if(!m||!logit)throw std::invalid_argument("null model/output");check_pcm(pcm,160000);
        *logit=m->impl->submit(pcm).get();error.clear();return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
int shield_predict_batch(shield_model* m,const float* pcm,size_t count,size_t stride,float* logits){
    try {if(!m||(!logits&&count)||(!pcm&&count)||stride<160000 ||count>1000000)throw std::invalid_argument("invalid batch arguments");
        if(count && stride>std::numeric_limits<size_t>::max()/count)throw std::invalid_argument("batch size overflow");
        for(size_t i=0;i<count;++i){logits[i]=missing_float;check_pcm(pcm+i*stride,160000);}
        // Bounded chunks avoid an unbounded scheduling queue for a large batch.
        for(size_t base=0;base<count;base+=64){
            const size_t end=std::min(count,base+64);
            std::vector<std::future<float>> futures;futures.reserve(end-base);
            try {for(size_t i=base;i<end;++i)futures.push_back(m->impl->submit(pcm+i*stride));}
            catch(...){for(auto& f:futures)f.wait();throw;}
            std::exception_ptr failure;
            for(size_t i=base;i<end;++i)try{logits[i]=futures[i-base].get();}catch(...){failure=std::current_exception();}
            if(failure)std::rethrow_exception(failure);
        }
        error.clear();return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
shield_stream* shield_stream_create(shield_model* m,const char* id,double threshold,const char* version,int k,int n){
    try {if(!id||!version||!std::isfinite(threshold)||threshold<0||threshold>1||k<1||n<k||n>100000)
            throw std::invalid_argument("invalid stream configuration");
        auto s=std::make_unique<shield_stream>();if(m)s->model=m->impl;
        s->id=id;s->version=version;s->threshold=threshold;s->alert_k=k;s->alert_n=n;s->recent.reserve(n);s->cache.trusted_shift=true;
        error.clear();return s.release();
    }catch(const std::exception& e){error=e.what();return nullptr;}
}
void shield_stream_free(shield_stream* s){if(!s)return;{std::unique_lock<std::mutex> lock(s->mutex);s->available.wait(lock,[s]{return !s->busy;});}delete s;}
void shield_stream_reset(shield_stream* s){if(!s)return;std::unique_lock<std::mutex> lock(s->mutex);s->available.wait(lock,[s]{return !s->busy;});
    s->filled=s->since=s->write=0;s->recent.clear();s->cache.reset();s->sequence_cache.reset();s->lp_state.fill(0.0);}
int shield_stream_set_lowpass(shield_stream* s,int enabled){
    try{if(!s)throw std::invalid_argument("null stream");if(enabled!=0&&enabled!=1)throw std::invalid_argument("lowpass must be 0 or 1");
        std::unique_lock<std::mutex> lock(s->mutex);s->available.wait(lock,[s]{return !s->busy;});
        if(s->filled||s->since)throw std::invalid_argument("lowpass can change only on an empty stream (after create or reset)");
        s->lowpass=enabled;s->lp_state.fill(0.0);
        if(enabled)s->lp_ring.assign(160000,0.0f);else{s->lp_ring.clear();s->lp_ring.shrink_to_fit();}
        s->cache.reset();s->sequence_cache.reset();error.clear();return 0;}
    catch(const std::exception& e){error=e.what();return -1;}
}
int shield_stream_lowpass(const shield_stream* s){return s?int(s->lowpass):-1;}
int shield_stream_set_cache_level(shield_stream* s,int level){
    try{if(!s)throw std::invalid_argument("null stream");std::unique_lock<std::mutex> lock(s->mutex);s->available.wait(lock,[s]{return !s->busy;});s->sequence_cache.set_level(level);return 0;}
    catch(const std::exception& e){error=e.what();return -1;}
}
const char* shield_stream_id(const shield_stream* s){return s?s->id.c_str():nullptr;}
const char* shield_stream_version(const shield_stream* s){return s?s->version.c_str():nullptr;}
size_t shield_stream_bytes(const shield_stream* s){if(!s)return 0;std::unique_lock<std::mutex> lock(s->mutex);const_cast<shield_stream*>(s)->available.wait(lock,[s]{return !s->busy;});
    return sizeof(*s)+(s->ring.capacity()+s->lp_ring.capacity())*sizeof(float)+s->cache.bytes()-sizeof(s->cache)+s->sequence_cache.bytes()-sizeof(s->sequence_cache)+s->recent.capacity()+s->id.capacity()+s->version.capacity();}
double shield_speech_seconds(const float* pcm,size_t samples){
    try{check_pcm(pcm,samples);return speech(pcm,samples);}catch(const std::exception& e){error=e.what();return std::numeric_limits<double>::quiet_NaN();}}
int shield_stream_push(shield_stream* s,const float* pcm,size_t samples,double t_end,
                       shield_result* results,size_t capacity,size_t* count){
    if(count)*count=0;
    try {if(!s||!count)throw std::invalid_argument("null stream/count");
        check_pcm(pcm,samples);
        if(std::isinf(t_end))throw std::invalid_argument("infinite timestamp");
        std::unique_lock<std::mutex> lock(s->mutex);s->available.wait(lock,[s]{return !s->busy;});
        if(samples>std::numeric_limits<size_t>::max()-s->since)throw std::invalid_argument("sample count overflow");
        size_t needed=(s->since+samples)/32000;
        if(capacity<needed||(!results&&needed))throw std::invalid_argument("insufficient result capacity");
        if(std::isnan(t_end))t_end=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        size_t consumed=0;
        while(consumed<samples){
            const size_t take=std::min({samples-consumed,32000-s->since,160000-s->write});
            std::memcpy(s->ring.data()+s->write,pcm+consumed,take*sizeof(float));
            if(s->lowpass)lowpass_run(s->lp_state,pcm+consumed,s->lp_ring.data()+s->write,take);
            s->write=(s->write+take)%160000;s->filled=std::min(size_t(160000),s->filled+take);
            s->since+=take;consumed+=take;
            if(s->since<32000)continue;
            s->since=0;
            shield_result r{};r.window_end=t_end-double(samples-consumed)/16000;
            r.window_start=r.window_end-10;r.score=r.logit=missing_float;r.alert=-1;r.status=SHIELD_COLLECTING;
            // At hop boundaries write is a multiple of the 320-sample VAD
            // frame. Circularly rotating whole frames preserves the percentile
            // and active-frame count, so VAD need not linearize the PCM.
            r.valid_speech_sec=speech(s->ring.data(),s->filled);
            if(s->filled==160000){
                if(r.valid_speech_sec<3.0)r.status=SHIELD_INSUFFICIENT_SPEECH;
                else if(!s->model)r.status=SHIELD_UNAVAILABLE;
                else try {
                    r.logit=s->model->submit(s->model_ring(),&s->cache,s->write,&s->sequence_cache).get();
                    r.score=1.0/(1.0+std::exp(-double(r.logit)));
                    r.status=SHIELD_VALID;
                    if(s->recent.size()==size_t(s->alert_n))s->recent.erase(s->recent.begin());
                    s->recent.push_back(r.score>=s->threshold);
                    r.alert=std::count(s->recent.begin(),s->recent.end(),true)>=s->alert_k;
                }catch(const std::exception& e){error=e.what();r.status=SHIELD_UNAVAILABLE;r.score=r.logit=missing_float;r.alert=-1;s->cache.reset();s->sequence_cache.reset();}
            }
            if(r.status!=SHIELD_VALID){s->cache.valid=false;s->sequence_cache.reset();}
            results[(*count)++]=r;
        }
        return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
uint32_t shield_stream_phase_samples(const shield_stream* s){
    if(!s)return 0;uint64_t h=1469598103934665603ULL;
    for(unsigned char c:s->id){h^=c;h*=1099511628211ULL;}return uint32_t(h%32000);
}
shield_ticket* shield_stream_submit_hop(shield_stream* s,const float* pcm,double t_end){
    try{
        if(!s)throw std::invalid_argument("null stream");check_pcm(pcm,32000);
        if(std::isinf(t_end))throw std::invalid_argument("infinite timestamp");
        if(std::isnan(t_end))t_end=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        auto ticket=std::make_unique<shield_ticket>();ticket->owner=s->model;ticket->timing=std::make_shared<TicketTiming>();auto timing=ticket->timing;
        auto promise=std::make_shared<std::promise<shield_result>>();ticket->future=promise->get_future();
        std::unique_lock<std::mutex> lock(s->mutex);s->available.wait(lock,[s]{return !s->busy;});
        if(s->since)throw std::invalid_argument("asynchronous hop must start at a completed hop boundary");
        std::copy_n(pcm,32000,s->ring.data()+s->write);if(s->lowpass)lowpass_run(s->lp_state,pcm,s->lp_ring.data()+s->write,32000);s->write=(s->write+32000)%160000;s->filled=std::min(size_t(160000),s->filled+32000);
        shield_result r{};r.window_end=t_end;r.window_start=t_end-10;r.score=r.logit=missing_float;r.alert=-1;
        r.valid_speech_sec=speech(s->ring.data(),s->filled);r.status=SHIELD_COLLECTING;
        if(s->filled==160000)r.status=r.valid_speech_sec<3?SHIELD_INSUFFICIENT_SPEECH:(s->model?SHIELD_VALID:SHIELD_UNAVAILABLE);
        if(r.status!=SHIELD_VALID){s->cache.valid=false;s->sequence_cache.reset();timing->started=timing->finished=std::chrono::steady_clock::now();promise->set_value(r);return ticket.release();}
        s->busy=true;
        try{s->model->enqueue([s,r,promise,timing]()mutable{
            timing->started=std::chrono::steady_clock::now();
            try{r.logit=s->model->run(s->model_ring(),&s->cache,s->write,&s->sequence_cache);
                r.score=1.0/(1.0+std::exp(-double(r.logit)));}
            catch(...){r.status=SHIELD_UNAVAILABLE;r.score=r.logit=missing_float;r.alert=-1;}
            {std::lock_guard<std::mutex> done(s->mutex);
                if(r.status==SHIELD_VALID){if(s->recent.size()==size_t(s->alert_n))s->recent.erase(s->recent.begin());s->recent.push_back(r.score>=s->threshold);r.alert=std::count(s->recent.begin(),s->recent.end(),true)>=s->alert_k;}
                else{s->cache.reset();s->sequence_cache.reset();}
                s->busy=false;s->available.notify_all();
            }
            timing->finished=std::chrono::steady_clock::now();promise->set_value(r);
        });}catch(...){s->busy=false;s->cache.reset();s->sequence_cache.reset();s->available.notify_all();throw;}
        return ticket.release();
    }catch(const std::exception& e){error=e.what();return nullptr;}
}
int shield_ticket_ready(shield_ticket* t){if(!t)return -1;return t->resolved||t->future.wait_for(std::chrono::seconds(0))==std::future_status::ready;}
int shield_ticket_wait(shield_ticket* t,shield_result* out){
    try{if(!t||!out)throw std::invalid_argument("null ticket/result");if(!t->resolved){t->result=t->future.get();t->resolved=true;}*out=t->result;return 0;}
    catch(const std::exception& e){error=e.what();return -1;}
}
void shield_ticket_free(shield_ticket* t){if(!t)return;if(!t->resolved)t->future.wait();delete t;}
int shield_ticket_timing(shield_ticket* t,double* queue_ms,double* compute_ms,double* completed_monotonic){
    if(!t||!queue_ms||!compute_ms||!completed_monotonic)return -1;
    if(!shield_ticket_ready(t))return 1;
    *queue_ms=std::chrono::duration<double,std::milli>(t->timing->started-t->timing->created).count();
    *compute_ms=std::chrono::duration<double,std::milli>(t->timing->finished-t->timing->started).count();
    *completed_monotonic=std::chrono::duration<double>(t->timing->finished.time_since_epoch()).count();return 0;
}
}
