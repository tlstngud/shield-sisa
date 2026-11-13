// Shared by prep_cli / prep_bench: ffprobe + ffmpeg subprocess decoding identical to reference/shim/av.py.
#pragma once
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern char** environ;

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// run argv, capture stdout; returns exit status (or -1)
static inline int run_capture(const std::vector<std::string>& args, std::string& out) {
    int pfd[2];
    if (pipe(pfd)) return -1;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, pfd[1], 1);
    posix_spawn_file_actions_addclose(&fa, pfd[0]);
    posix_spawn_file_actions_addclose(&fa, pfd[1]);
    std::vector<char*> av;
    for (auto& a : args) av.push_back(const_cast<char*>(a.c_str()));
    av.push_back(nullptr);
    pid_t pid;
    int rc = posix_spawnp(&pid, av[0], &fa, nullptr, av.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(pfd[1]);
    if (rc) {
        close(pfd[0]);
        return -1;
    }
    out.clear();
    std::vector<char> buf(1 << 20);
    ssize_t n;
    while ((n = read(pfd[0], buf.data(), buf.size())) > 0) out.append(buf.data(), (size_t)n);
    close(pfd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

struct Probe {
    int64_t tb_num = 0, tb_den = 0;
    bool has_duration_ts = false;
    int64_t duration_ts = 0;
    bool has_container = false;
    long long container_us = 0;
    int width = 0, height = 0;
    std::vector<int64_t> pts;
};

static inline bool probe(const std::string& path, Probe& p, std::string& err) {
    std::string out;
    if (run_capture({"ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream=time_base,duration_ts,width,height:format=duration", "-of", "default=noprint_wrappers=1",
                     path},
                    out) != 0) {
        err = "ffprobe stream failed";
        return false;
    }
    size_t i = 0;
    while (i < out.size()) {
        size_t e = out.find('\n', i);
        if (e == std::string::npos) e = out.size();
        std::string line = out.substr(i, e - i);
        i = e + 1;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "time_base") {
            size_t sl = v.find('/');
            p.tb_num = atoll(v.substr(0, sl).c_str());
            p.tb_den = atoll(v.substr(sl + 1).c_str());
        } else if (k == "duration_ts") {
            if (v != "N/A" && !v.empty()) p.has_duration_ts = true, p.duration_ts = atoll(v.c_str());
        } else if (k == "width") {
            p.width = atoi(v.c_str());
        } else if (k == "height") {
            p.height = atoi(v.c_str());
        } else if (k == "duration") {
            if (v != "N/A" && !v.empty()) {
                // int(round(float(duration) * 1000000)) : Python round half to even
                p.has_container = true;
                p.container_us = (long long)std::nearbyint(strtod(v.c_str(), nullptr) * 1000000);
            }
        }
    }
    if (run_capture({"ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,flags", "-of",
                     "csv=p=0", path},
                    out) != 0) {
        err = "ffprobe packets failed";
        return false;
    }
    i = 0;
    while (i < out.size()) {
        size_t e = out.find('\n', i);
        if (e == std::string::npos) e = out.size();
        std::string line = out.substr(i, e - i);
        i = e + 1;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        size_t c = line.find(',');
        std::string ps = line.substr(0, c), flags = c == std::string::npos ? "" : line.substr(c + 1);
        size_t c2 = flags.find(',');
        if (c2 != std::string::npos) flags = flags.substr(0, c2);
        if (ps == "N/A" || ps.empty() || flags.find('D') != std::string::npos) continue;
        p.pts.push_back(atoll(ps.c_str()));
    }
    std::sort(p.pts.begin(), p.pts.end());
    p.pts.erase(std::unique(p.pts.begin(), p.pts.end()), p.pts.end());
    if (p.tb_den <= 0 || p.width <= 0 || p.height <= 0) {
        err = "ffprobe: missing stream fields";
        return false;
    }
    return true;
}

static inline bool decode(const std::string& path, const std::vector<int64_t>& want, bool yuv, int w, int h,
                   std::string& raw, std::string& err) {
    std::string expr;
    for (size_t i = 0; i < want.size(); i++) {
        if (i) expr += "+";
        expr += "eq(pts\\," + std::to_string(want[i]) + ")";
    }
    std::string vf = yuv ? "select='" + expr + "'"
                         : "select='" + expr + "',scale=in_color_matrix=bt601:in_range=auto:flags=bilinear,format=rgb24";
    std::vector<std::string> cmd = {"ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-copyts", "-i", path,
                                    "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf, "-fps_mode", "passthrough",
                                    "-f", "rawvideo", "-pix_fmt", yuv ? "yuv420p" : "rgb24", "pipe:1"};
    if (run_capture(cmd, raw) != 0) {
        err = "ffmpeg failed";
        return false;
    }
    size_t fb = yuv ? (size_t)w * h + 2 * (size_t)((w + 1) / 2) * ((h + 1) / 2) : (size_t)w * h * 3;
    if (raw.size() != fb * want.size()) {
        err = "decoded " + std::to_string(raw.size() / fb) + " frames, wanted " + std::to_string(want.size());
        return false;
    }
    return true;
}

