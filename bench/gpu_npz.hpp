// Minimal reader for the reference crops: numpy .npz (zip, deflate or stored) holding frames.npy uint8 [20,180,320,3].
// Used only by the bench / verification tools (never by the engine).
#pragma once
#include <zlib.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace shvbench {

inline uint32_t rd32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
inline uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | p[1] << 8); }

// Returns the raw bytes of member `member` of the zip archive at `path`.
inline std::vector<uint8_t> zip_member(const std::string& path, const std::string& member) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> z((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (z.size() < 22) throw std::runtime_error("not a zip: " + path);
    size_t eocd = std::string::npos;
    for (size_t i = z.size() - 22 + 1; i-- > 0;)
        if (rd32(&z[i]) == 0x06054b50u) { eocd = i; break; }
    if (eocd == std::string::npos) throw std::runtime_error("no zip directory: " + path);
    const uint16_t entries = rd16(&z[eocd + 10]);
    size_t cd = rd32(&z[eocd + 16]);
    for (uint16_t e = 0; e < entries; ++e) {
        if (cd + 46 > z.size() || rd32(&z[cd]) != 0x02014b50u) throw std::runtime_error("bad zip directory: " + path);
        const uint16_t method = rd16(&z[cd + 10]);
        uint64_t csize = rd32(&z[cd + 20]), usize = rd32(&z[cd + 24]);
        const uint16_t nlen = rd16(&z[cd + 28]), xlen = rd16(&z[cd + 30]), clen = rd16(&z[cd + 32]);
        uint64_t lho = rd32(&z[cd + 42]);
        const std::string name(reinterpret_cast<const char*>(&z[cd + 46]), nlen);
        if (csize == 0xffffffffu || usize == 0xffffffffu || lho == 0xffffffffu) {  // zip64 extra field
            size_t x = cd + 46 + nlen, xe = x + xlen;
            while (x + 4 <= xe) {
                const uint16_t id = rd16(&z[x]), sz = rd16(&z[x + 2]);
                if (id == 1) {
                    size_t q = x + 4;
                    auto r64 = [&](size_t at) { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = v << 8 | z[at + i]; return v; };
                    if (usize == 0xffffffffu) { usize = r64(q); q += 8; }
                    if (csize == 0xffffffffu) { csize = r64(q); q += 8; }
                    if (lho == 0xffffffffu) { lho = r64(q); q += 8; }
                }
                x += 4 + sz;
            }
        }
        if (name == member) {
            if (lho + 30 > z.size() || rd32(&z[lho]) != 0x04034b50u) throw std::runtime_error("bad local header: " + path);
            const size_t data = lho + 30 + rd16(&z[lho + 26]) + rd16(&z[lho + 28]);
            if (data + csize > z.size()) throw std::runtime_error("truncated zip: " + path);
            std::vector<uint8_t> out(usize);
            if (method == 0) {
                std::memcpy(out.data(), &z[data], usize);
            } else if (method == 8) {
                z_stream s{};
                if (inflateInit2(&s, -15) != Z_OK) throw std::runtime_error("inflateInit2");
                s.next_in = &z[data]; s.avail_in = uInt(csize);
                s.next_out = out.data(); s.avail_out = uInt(usize);
                const int rc = inflate(&s, Z_FINISH);
                inflateEnd(&s);
                if (rc != Z_STREAM_END || s.total_out != usize) throw std::runtime_error("inflate failed: " + path);
            } else {
                throw std::runtime_error("unsupported zip method: " + path);
            }
            return out;
        }
        cd += 46 + nlen + xlen + clen;
    }
    throw std::runtime_error("member " + member + " not in " + path);
}

// frames.npy of a reference crop -> 20*180*320*3 bytes (C order, uint8).
inline std::vector<uint8_t> load_crop(const std::string& path) {
    std::vector<uint8_t> npy = zip_member(path, "frames.npy");
    if (npy.size() < 10 || std::memcmp(npy.data(), "\x93NUMPY", 6)) throw std::runtime_error("bad npy in " + path);
    const int major = npy[6];
    size_t hlen, hoff;
    if (major == 1) { hlen = rd16(&npy[8]); hoff = 10; }
    else { hlen = rd32(&npy[8]); hoff = 12; }
    const std::string hdr(reinterpret_cast<const char*>(&npy[hoff]), hlen);
    if (hdr.find("'|u1'") == std::string::npos || hdr.find("(20, 180, 320, 3)") == std::string::npos ||
        hdr.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("unexpected crop array header in " + path + ": " + hdr);
    const size_t n = size_t(20) * 180 * 320 * 3;
    if (npy.size() != hoff + hlen + n) throw std::runtime_error("bad crop size in " + path);
    return std::vector<uint8_t>(npy.begin() + long(hoff + hlen), npy.end());
}

// SHA-256 hex of a string (crop file stem = sha256(sample_id)[:24]).
inline std::string sha256_hex(const std::string& msg) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> m(msg.begin(), msg.end());
    const uint64_t bits = uint64_t(m.size()) * 8;
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 7; i >= 0; --i) m.push_back(uint8_t(bits >> (i * 8)));
    auto rr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = uint32_t(m[off + 4 * i]) << 24 | uint32_t(m[off + 4 * i + 1]) << 16 | uint32_t(m[off + 4 * i + 2]) << 8 | m[off + 4 * i + 3];
        for (int i = 16; i < 64; ++i)
            w[i] = w[i - 16] + (rr(w[i - 15], 7) ^ rr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] + (rr(w[i - 2], 17) ^ rr(w[i - 2], 19) ^ (w[i - 2] >> 10));
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], z = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = z + (rr(e, 6) ^ rr(e, 11) ^ rr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rr(a, 2) ^ rr(a, 13) ^ rr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            z = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += z;
    }
    static const char* hx = "0123456789abcdef";
    std::string out;
    for (uint32_t x : h)
        for (int i = 7; i >= 0; --i) out += hx[(x >> (i * 4)) & 15];
    return out;
}

}  // namespace shvbench
