// Helpers shared by the service tests / benchmarks / demo (never used by libshield_av itself):
// streaming ffmpeg decode identical to reference/shim/av.py (BT.601, bilinear swscale, presentation order),
// RTFS map / published-results parsing, win_v2 int16 audio windows (mmap'ed .npy), small stats helpers.
#pragma once
#include "data_path.hpp"
#include "gpu_npz.hpp"       // bench/: reference crop .npz reader, sha256_hex
#include "prep_decode.hpp"   // tools/: run_capture, probe() (ffprobe exactly like the shim)
#include "shield_av.h"
#include "shv_prep.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace svc {

inline double now_s() { return shav_monotonic(); }

// ------------------------------------------------------------------------------------------------ JSON (flat lines)
inline bool jstr(const std::string& line, const std::string& key, std::string& out) {
    const std::string k = "\"" + key + "\": \"";
    size_t p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    size_t q = p;
    while (q < line.size() && line[q] != '"') q += line[q] == '\\' ? 2 : 1;
    out = line.substr(p, q - p);
    return true;
}
inline bool jnum(const std::string& line, const std::string& key, double& out) {
    const std::string k = "\"" + key + "\": ";
    size_t p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    if (line.compare(p, 4, "null") == 0) { out = NAN; return true; }
    out = std::strtod(line.c_str() + p, nullptr);
    return true;
}
inline bool jarr(const std::string& line, const std::string& key, std::vector<double>& out) {
    const std::string k = "\"" + key + "\": [";
    size_t p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    size_t q = line.find(']', p);
    out.clear();
    std::stringstream ss(line.substr(p, q - p));
    std::string x;
    while (std::getline(ss, x, ',')) out.push_back(std::strtod(x.c_str(), nullptr));
    return true;
}

struct MapRow {
    std::string sample_id, split, method, path, source_group;
    int label = 0;
    double window_start = 0;
    // published reference results
    int pub_anchors = -1;
    std::vector<double> pub_times, pub_frame_ids;
    double pub_score = NAN;
};

inline std::vector<MapRow> load_map(const std::string& split) {
    std::vector<MapRow> rows;
    std::ifstream in(shvbench_data("/shieldv_codec/maps/orig.jsonl"));
    std::string line;
    while (std::getline(in, line)) {
        MapRow r;
        double v;
        jstr(line, "sample_id", r.sample_id);
        jstr(line, "split", r.split);
        jstr(line, "method", r.method);
        jstr(line, "path", r.path);
        jstr(line, "source_group", r.source_group);
        if (jnum(line, "label", v)) r.label = int(v);
        jnum(line, "window_start", r.window_start);
        if (split == "all" || r.split == split) rows.push_back(r);
    }
    std::map<std::string, size_t> idx;
    for (size_t i = 0; i < rows.size(); ++i) idx[rows[i].sample_id] = i;
    std::ifstream pr(shvbench_data("/shieldv_codec/results/orig_bt601.jsonl"));
    while (std::getline(pr, line)) {
        std::string id;
        jstr(line, "sample_id", id);
        auto it = idx.find(id);
        if (it == idx.end()) continue;
        MapRow& r = rows[it->second];
        double v;
        if (jnum(line, "anchors", v)) r.pub_anchors = int(v);
        jnum(line, "score", r.pub_score);
        jarr(line, "times", r.pub_times);
        jarr(line, "frame_ids", r.pub_frame_ids);
    }
    return rows;
}

inline std::string crop_path(const std::string& sample_id) {
    return shvbench_data("/shieldv_codec/crops/orig_bt601/") + shvbench::sha256_hex(sample_id).substr(0, 24) + ".npz";
}

// ------------------------------------------------------------------------------------------------ streaming decode
// ffmpeg writes every decoded frame (presentation order) as rgb24 (shim colour path) or native yuv420p.
struct FrameReader {
    pid_t pid = -1;
    int fd = -1;
    int w = 0, h = 0;
    bool yuv = false;
    size_t frame_bytes = 0;
    bool open(const std::string& path, bool yuv_, int w_, int h_, int threads = 2) {
        yuv = yuv_;
        w = w_;
        h = h_;
        frame_bytes = yuv ? size_t(w) * h + 2 * size_t((w + 1) / 2) * ((h + 1) / 2) : size_t(w) * h * 3;
        std::vector<std::string> args = {"ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-copyts", "-threads",
                                         std::to_string(threads), "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn"};
        if (!yuv) {
            args.push_back("-vf");
            args.push_back("scale=in_color_matrix=bt601:in_range=auto:flags=bilinear,format=rgb24");
        }
        for (const char* a : {"-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt"}) args.push_back(a);
        args.push_back(yuv ? "yuv420p" : "rgb24");
        args.push_back("pipe:1");
        int pfd[2];
        // O_CLOEXEC: concurrently spawned decoders must not inherit each other's pipe ends (a stray read end keeps a
        // writer blocked after the reader closed early)
        if (pipe2(pfd, O_CLOEXEC)) return false;
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, pfd[1], 1);
        posix_spawn_file_actions_addclose(&fa, pfd[0]);
        posix_spawn_file_actions_addclose(&fa, pfd[1]);
        // stderr off: closing the pipe early makes ffmpeg report EPIPE; real failures show up as missing frames
        posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
        std::vector<char*> av;
        for (auto& a : args) av.push_back(const_cast<char*>(a.c_str()));
        av.push_back(nullptr);
        const int rc = posix_spawnp(&pid, av[0], &fa, nullptr, av.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        close(pfd[1]);
        if (rc) {
            close(pfd[0]);
            pid = -1;
            return false;
        }
        fd = pfd[0];
        return true;
    }
    // false at end of stream
    bool next(uint8_t* buf) {
        size_t got = 0;
        while (got < frame_bytes) {
            const ssize_t n = read(fd, buf + got, frame_bytes - got);
            if (n <= 0) return false;
            got += size_t(n);
        }
        return true;
    }
    // closing the pipe early ends ffmpeg (SIGPIPE / EPIPE on its next write); returns its exit status
    int close_wait() {
        if (fd >= 0) ::close(fd);
        fd = -1;
        int st = 0;
        if (pid > 0) waitpid(pid, &st, 0);
        pid = -1;
        return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    }
};

struct Clip {
    std::string sample_id;
    int w = 0, h = 0;
    bool yuv = true;
    size_t frame_bytes = 0;
    std::vector<double> t;        // seconds after the first frame (exact reference times)
    std::vector<uint8_t> data;    // frames back to back
    const uint8_t* frame(size_t i) const { return data.data() + i * frame_bytes; }
};

// Decodes up to max_seconds of a video into memory (presentation order, times from ffprobe packets).
inline bool decode_clip(const std::string& path, bool yuv, double max_seconds, Clip& c, std::string& err) {
    Probe p;
    if (!probe(path, p, err)) return false;
    c.w = p.width;
    c.h = p.height;
    c.yuv = yuv;
    FrameReader fr;
    if (!fr.open(path, yuv, p.width, p.height)) {
        err = "ffmpeg spawn failed";
        return false;
    }
    c.frame_bytes = fr.frame_bytes;
    const double origin = shv_pts_seconds(p.pts[0], p.tb_num, p.tb_den);
    size_t n = 0;
    for (; n < p.pts.size(); ++n) {
        const double t = shv_pts_seconds(p.pts[n], p.tb_num, p.tb_den) - origin;
        if (t > max_seconds) break;
        c.data.resize((n + 1) * c.frame_bytes);
        if (!fr.next(c.data.data() + n * c.frame_bytes)) break;
        c.t.push_back(t);
    }
    fr.close_wait();
    c.data.resize(n * c.frame_bytes);
    c.t.resize(n);
    return n > 0;
}

inline int push_frame(shav_session* s, double t, const uint8_t* f, int w, int h, bool yuv) {
    if (!yuv) return shav_session_push_video_rgb(s, t, f, w, h, ptrdiff_t(3) * w);
    const uint8_t* y = f;
    const uint8_t* u = y + size_t(w) * h;
    const int cw = (w + 1) / 2, ch = (h + 1) / 2;
    const uint8_t* v = u + size_t(cw) * ch;
    return shav_session_push_video_i420(s, t, y, w, u, cw, v, cw, w, h);
}

// ------------------------------------------------------------------------------------------------ win_v2 audio
struct Npy16 {
    const int16_t* data = nullptr;
    size_t rows = 0, cols = 0;
    void* map = nullptr;
    size_t map_len = 0;
    bool open(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        fstat(fd, &st);
        map_len = size_t(st.st_size);
        map = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (map == MAP_FAILED) return false;
        const uint8_t* b = static_cast<const uint8_t*>(map);
        if (std::memcmp(b, "\x93NUMPY", 6)) return false;
        size_t hlen, hoff;
        if (b[6] == 1) hlen = size_t(b[8] | b[9] << 8), hoff = 10;
        else hlen = size_t(b[8] | b[9] << 8 | b[10] << 16 | size_t(b[11]) << 24), hoff = 12;
        const std::string hdr(reinterpret_cast<const char*>(b + hoff), hlen);
        if (hdr.find("<i2") == std::string::npos || hdr.find("False") == std::string::npos) return false;
        const size_t sp = hdr.find("'shape': (");
        rows = std::strtoull(hdr.c_str() + sp + 10, nullptr, 10);
        cols = std::strtoull(hdr.c_str() + hdr.find(',', sp) + 1, nullptr, 10);
        data = reinterpret_cast<const int16_t*>(b + hoff + hlen);
        return true;
    }
    const int16_t* row(size_t i) const { return data + i * cols; }
};

inline std::vector<int> load_labels(const std::string& jsonl) {
    std::vector<int> out;
    std::ifstream in(jsonl);
    std::string line;
    while (std::getline(in, line)) {
        double v = -1;
        jnum(line, "label", v);
        out.push_back(int(v));
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ stats
struct Pct {
    double p50 = NAN, p95 = NAN, p99 = NAN, mean = NAN, max = NAN;
    size_t n = 0;
};
inline Pct pct(std::vector<double> v) {
    Pct p;
    p.n = v.size();
    if (v.empty()) return p;
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[std::min(v.size() - 1, size_t(std::ceil(q * double(v.size())) - 1))]; };
    double s = 0;
    for (double x : v) s += x;
    p.p50 = at(0.50);
    p.p95 = at(0.95);
    p.p99 = at(0.99);
    p.mean = s / double(v.size());
    p.max = v.back();
    return p;
}
inline std::string fmt(double v, const char* f = "%.3f") {
    if (!std::isfinite(v)) return "null";
    char b[64];
    std::snprintf(b, sizeof b, f, v);
    return b;
}
inline std::string read_file(const std::string& p) {
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
inline size_t rss_bytes() {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("VmRSS:", 0) == 0) return size_t(std::strtoull(line.c_str() + 6, nullptr, 10)) * 1024;
    return 0;
}
inline size_t hwm_bytes() {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("VmHWM:", 0) == 0) return size_t(std::strtoull(line.c_str() + 6, nullptr, 10)) * 1024;
    return 0;
}
// CPU seconds per thread name (comm) of this process
inline std::map<std::string, double> thread_cpu() {
    std::map<std::string, double> out;
    const long tck = sysconf(_SC_CLK_TCK);
    DIR* d = opendir("/proc/self/task");
    if (!d) return out;
    while (dirent* de = readdir(d)) {
        if (de->d_name[0] == '.') continue;
        const std::string s = read_file(std::string("/proc/self/task/") + de->d_name + "/stat");
        const size_t a = s.find('('), b = s.rfind(')');
        if (a == std::string::npos || b == std::string::npos) continue;
        const std::string comm = s.substr(a + 1, b - a - 1);
        std::stringstream ss(s.substr(b + 2));
        std::string tok;
        std::vector<std::string> f;
        while (ss >> tok) f.push_back(tok);
        if (f.size() < 13) continue;
        out[comm] += double(std::strtoull(f[11].c_str(), nullptr, 10) + std::strtoull(f[12].c_str(), nullptr, 10)) / double(tck);
    }
    closedir(d);
    return out;
}
inline double process_cpu_s() {
    rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return double(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) + 1e-6 * double(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
}

}  // namespace svc
