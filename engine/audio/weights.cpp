#include "core.hpp"
#include "sha256.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstring>

namespace shield {
namespace {
template<class T> T read(std::ifstream& in) {
    T x{}; in.read(reinterpret_cast<char*>(&x), sizeof x);
    if (!in) throw std::runtime_error("truncated weights");
    return x;
}
}
Weights::Weights(const std::string& path) {
    static_assert(sizeof(float)==4, "float32 required");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open weights: "+path);
    char magic[8]; in.read(magic,8);
    if (!in || std::memcmp(magic,"SHLDA001",8)) throw std::runtime_error("invalid weight format");
    auto count=read<uint32_t>(in);
    if (!count || count>10000) throw std::runtime_error("invalid tensor count");
    uint64_t total=0;
    for (uint32_t i=0;i<count;++i) {
        auto n=read<uint32_t>(in);
        if (!n || n>4096) throw std::runtime_error("invalid tensor name length");
        std::string name(n,'\0'); in.read(name.data(),n);
        auto rank=read<uint32_t>(in);
        if (rank>8) throw std::runtime_error("invalid tensor rank");
        Tensor t; uint64_t elements=1;
        for (uint32_t d=0;d<rank;++d) {
            auto dim=read<uint32_t>(in);
            if (!dim || dim>10000000 || elements>100000000/dim)
                throw std::runtime_error("invalid tensor shape");
            t.shape.push_back(static_cast<int>(dim)); elements*=dim;
        }
        if (read<uint64_t>(in)!=elements || (total+=elements)>100000000)
            throw std::runtime_error("invalid tensor size");
        t.data.resize(elements);
        in.read(reinterpret_cast<char*>(t.data.data()),elements*sizeof(float));
        if (!in) throw std::runtime_error("truncated tensor: "+name);
        if (!tensors.emplace(name,std::move(t)).second)
            throw std::runtime_error("duplicate tensor: "+name);
    }
    if (in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing weight data");
    in.clear();in.seekg(0);fingerprint=sha256_stream(in);
}
std::size_t Weights::bytes() const {
    std::size_t n=0; for (const auto& item:tensors) n+=item.second.data.size()*sizeof(float);
    return n;
}
}
