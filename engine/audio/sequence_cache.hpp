#pragma once
#include "core.hpp"
#include <array>
#include <stdexcept>
#include <vector>

namespace shield {
// Participant-owned exact GDN-A preprocessing. The caller must pass true only
// after the frontend proves a single aligned hop (new tokens[5,95) equal old
// tokens[30,120)), and must reset after skipped/failed hops. No recurrent state
// is retained. Immutable model identity and numeric mode are checked internally.
struct SequenceCache {
    int level=1; // 0 off, 1 six projections, 2 also eligible Q/K/V preprocessing.
    bool valid=false;
    const Weights* weights=nullptr;
    std::uint64_t execution_mode=0;
    int cached_level=0;
    std::array<std::vector<float>,6> projected;
    std::array<std::vector<float>,3> processed;
    std::size_t reused_projection_rows=0,reused_conv_rows=0;
    void reset(){valid=false;weights=nullptr;reused_projection_rows=reused_conv_rows=0;}
    void set_level(int value){
        if(value<0||value>2)throw std::invalid_argument("sequence cache level must be 0, 1 or 2");
        if(value!=level){reset();level=value;}
        if(value==0)for(auto& v:projected)std::vector<float>().swap(v);
        if(value<2)for(auto& v:processed)std::vector<float>().swap(v);
    }
    std::size_t bytes()const{
        std::size_t result=sizeof(*this);for(const auto& v:projected)result+=v.capacity()*sizeof(float);
        for(const auto& v:processed)result+=v.capacity()*sizeof(float);return result;
    }
};
void sequence_cached(const Weights&,const float* tokens,float& logit,SequenceCache*,bool reuse_frontend);
void sequence_cached_profile(const Weights&,const float* tokens,float& logit,SequenceCache*,bool reuse_frontend,double stage_ms[4]);
void release_sequence_weights(const Weights&);
std::size_t sequence_shared_bytes(const Weights&);
}
