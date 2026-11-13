// SHIELD-V CUDA engine: C API, scheduler worker, certificates.  See engine/include/shv_cuda.h for the contract.
#include "shv_cuda.h"
#include "engine.hpp"
#include "kernels.cuh"
#include "shv_cuda_revision.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
thread_local std::string g_err;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr size_t FB = SHV_CUDA_FRAME_BYTES, TF = SHV_CUDA_TOKEN_FLOATS;

bool is_pinned(const void* p, size_t bytes) {
    if (!p || !bytes) return false;
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return false; }
    if (a.type != cudaMemoryTypeHost) return false;
    cudaPointerAttributes b{};
    if (cudaPointerGetAttributes(&b, static_cast<const uint8_t*>(p) + bytes - 1) != cudaSuccess) { cudaGetLastError(); return false; }
    return b.type == cudaMemoryTypeHost;
}

struct Core;
}  // namespace

struct shv_cuda_job {
    // request (index arrays copied)
    const uint8_t* frames = nullptr;
    size_t n_frames = 0;
    float* tokens_out = nullptr;
    std::vector<int32_t> store_slots;
    size_t n_windows = 0;
    std::vector<int32_t> wframes, wslots;
    std::vector<const float*> wtokens;
    float* logits_out = nullptr;
    shv_cuda_callback cb = nullptr;
    void* user = nullptr;
    bool frames_pinned = false, tokens_out_pinned = false;
    int host_tokens = 0;
    // batch bookkeeping
    int frame_off = 0, window_off = 0;
    // completion
    mutable std::mutex m;
    std::condition_variable cv;
    int state = 0;          // 0 pending, 1 done, -1 failed
    bool released = false;  // callback returned
    std::string error;
    Clock::time_point t_submit, t_start, t_done;
    float gpu_ms = 0.0f;
    int batch_frames = 0, batch_windows = 0;
};

namespace {

struct SlotHost {
    uint8_t* frames = nullptr;       // pinned staging [max_frames][FB]
    float* tok_stage = nullptr;      // pinned [max_host_tokens][TF]
    const float** tokptr = nullptr;  // pinned [max_windows*20]
    int32_t* store = nullptr;        // pinned [max_frames]
    int32_t* d_store = nullptr;      // device [max_frames]
    float* tokens_out = nullptr;     // pinned [max_frames][TF]
    float* logits = nullptr;         // pinned [max_windows]
    cudaEvent_t h2d = nullptr, start = nullptr, end = nullptr;
};

struct Batch {
    int slot = 0;
    std::vector<shv_cuda_job*> jobs;
    int F = 0, B = 0;
    bool failed = false;
    std::string error;
    Clock::time_point t_start;
};

struct Core {
    std::unique_ptr<shv::Engine> eng;
    shv::HostModel hm;
    std::string mode, fingerprint;
    double threshold = 0.0;
    int device = 0;
    shv_cuda_options opt{};
    cudaStream_t s_comp = nullptr, s_copy = nullptr;
    std::vector<SlotHost> sh;
    float* d_pool = nullptr;
    size_t device_bytes = 0;

    std::mutex mu;
    std::condition_variable cv;
    std::deque<shv_cuda_job*> q;
    bool stopping = false;
    std::thread worker;

    ~Core() {
        for (auto& h : sh) {
            if (h.frames) cudaFreeHost(h.frames);
            if (h.tok_stage) cudaFreeHost(h.tok_stage);
            if (h.tokptr) cudaFreeHost(h.tokptr);
            if (h.store) cudaFreeHost(h.store);
            if (h.d_store) cudaFree(h.d_store);
            if (h.tokens_out) cudaFreeHost(h.tokens_out);
            if (h.logits) cudaFreeHost(h.logits);
            if (h.h2d) cudaEventDestroy(h.h2d);
            if (h.start) cudaEventDestroy(h.start);
            if (h.end) cudaEventDestroy(h.end);
        }
        if (d_pool) cudaFree(d_pool);
        eng.reset();
        if (s_comp) cudaStreamDestroy(s_comp);
        if (s_copy) cudaStreamDestroy(s_copy);
    }

    void init() {
        SHV_CUDA_CHECK(cudaSetDevice(device));
        SHV_CUDA_CHECK(cudaStreamCreateWithFlags(&s_comp, cudaStreamNonBlocking));
        SHV_CUDA_CHECK(cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking));
        shv::EngineConfig ec;
        ec.max_frames = opt.max_frames;
        ec.max_windows = opt.max_windows;
        ec.use_graphs = opt.use_graphs != 0;
        ec.slots = 2;
        shv::Precision p;
        if (!shv::parse_precision(mode, &p)) throw std::runtime_error("unknown mode: " + mode);
        eng = shv::make_engine(hm, p, ec);
        sh.resize(size_t(ec.slots));
        const size_t Fm = size_t(opt.max_frames), Bm = size_t(opt.max_windows), Hm = size_t(opt.max_host_tokens);
        for (auto& h : sh) {
            SHV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h.frames), Fm * FB, cudaHostAllocDefault));
            SHV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h.tok_stage), std::max<size_t>(Hm, 1) * TF * 4, cudaHostAllocDefault));
            SHV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h.tokptr), Bm * 20 * sizeof(float*), cudaHostAllocDefault));
            SHV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h.store), Fm * 4, cudaHostAllocDefault));
            SHV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&h.d_store), Fm * 4));
            SHV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h.tokens_out), Fm * TF * 4, cudaHostAllocDefault));
            SHV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h.logits), Bm * 4, cudaHostAllocDefault));
            SHV_CUDA_CHECK(cudaEventCreateWithFlags(&h.h2d, cudaEventDisableTiming));
            SHV_CUDA_CHECK(cudaEventCreate(&h.start));
            SHV_CUDA_CHECK(cudaEventCreate(&h.end));
        }
        if (opt.token_slots > 0) {
            SHV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pool), size_t(opt.token_slots) * TF * 4));
            SHV_CUDA_CHECK(cudaMemset(d_pool, 0, size_t(opt.token_slots) * TF * 4));
        }
        device_bytes = eng->weight_bytes() + eng->activation_bytes() + size_t(ec.slots) * (eng->slot_bytes() + Fm * 4) +
                       size_t(std::max(opt.token_slots, 0)) * TF * 4;
        SHV_CUDA_CHECK(cudaDeviceSynchronize());
        worker = std::thread([this] { run(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stopping = true;
        }
        cv.notify_all();
        if (worker.joinable()) worker.join();
    }

    // ---------------------------------------------------------------- completion
    static void complete(shv_cuda_job* j, bool ok, const std::string& err, float gpu_ms, int bf, int bw) {
        if (!ok) {
            for (size_t i = 0; i < j->n_windows; ++i) j->logits_out[i] = kNaN;
            if (j->tokens_out) std::fill(j->tokens_out, j->tokens_out + j->n_frames * TF, kNaN);
        }
        shv_cuda_callback cb;
        void* user;
        {
            std::lock_guard<std::mutex> lk(j->m);
            j->state = ok ? 1 : -1;
            j->error = ok ? std::string() : err;
            j->t_done = Clock::now();
            j->gpu_ms = gpu_ms;
            j->batch_frames = bf;
            j->batch_windows = bw;
            cb = j->cb;
            user = j->user;
            // Notify while holding j->m (here and below): shv_cuda_job_free waits for `released` under j->m and then
            // deletes the job, so a notify after the unlock could touch a destroyed condition variable.
            j->cv.notify_all();
        }
        if (cb) cb(j, ok ? 1 : -1, user);
        {
            std::lock_guard<std::mutex> lk(j->m);
            j->released = true;
            j->cv.notify_all();   // last access to *j by this thread
        }
    }

    // ---------------------------------------------------------------- batching
    // Take queued jobs (FIFO) while the batch limits hold and no job stores into a slot read or stored earlier.
    void take(std::vector<shv_cuda_job*>& out) {
        int F = 0, B = 0, Hn = 0;
        std::unordered_set<int32_t> read, stored;
        while (!q.empty()) {
            shv_cuda_job* j = q.front();
            if (!out.empty()) {
                if (F + int(j->n_frames) > opt.max_frames || B + int(j->n_windows) > opt.max_windows ||
                    Hn + j->host_tokens > opt.max_host_tokens)
                    break;
                bool hazard = false;
                for (int32_t s : j->store_slots)
                    if (s >= 0 && (read.count(s) || stored.count(s))) { hazard = true; break; }
                if (hazard) break;
            }
            q.pop_front();
            out.push_back(j);
            F += int(j->n_frames);
            B += int(j->n_windows);
            Hn += j->host_tokens;
            for (int32_t s : j->store_slots)
                if (s >= 0) stored.insert(s);
            for (size_t e = 0; e < j->wslots.size(); ++e)
                if ((j->wframes.empty() || j->wframes[e] < 0) && j->wslots[e] >= 0) read.insert(j->wslots[e]);
        }
    }

    Batch launch(int slot, std::vector<shv_cuda_job*>& jobs) {
        Batch b;
        b.slot = slot;
        b.jobs = jobs;
        b.t_start = Clock::now();
        SlotHost& h = sh[size_t(slot)];
        const shv::Engine::Slot& d = eng->slot(slot);
        try {
            int F = 0, B = 0, Hn = 0;
            bool any_store = false;
            // frames: pinned sources DMA directly, the rest is staged through the slot's pinned buffer
            int run0 = -1, run1 = -1;  // staged run [run0, run1) in frames
            auto flush = [&] {
                if (run0 >= 0 && run1 > run0)
                    SHV_CUDA_CHECK(cudaMemcpyAsync(d.d_frames + size_t(run0) * FB, h.frames + size_t(run0) * FB,
                                                   size_t(run1 - run0) * FB, cudaMemcpyHostToDevice, s_copy));
                run0 = run1 = -1;
            };
            for (shv_cuda_job* j : jobs) {
                j->t_start = b.t_start;
                j->frame_off = F;
                j->window_off = B;
                if (j->n_frames) {
                    if (j->frames_pinned) {
                        flush();
                        SHV_CUDA_CHECK(cudaMemcpyAsync(d.d_frames + size_t(F) * FB, j->frames, j->n_frames * FB,
                                                       cudaMemcpyHostToDevice, s_copy));
                    } else {
                        std::memcpy(h.frames + size_t(F) * FB, j->frames, j->n_frames * FB);
                        if (run0 < 0) run0 = F;
                        run1 = F + int(j->n_frames);
                    }
                    for (size_t f = 0; f < j->n_frames; ++f) {
                        const int32_t s = j->store_slots.empty() ? -1 : j->store_slots[f];
                        h.store[F + int(f)] = s;
                        any_store |= s >= 0;
                    }
                }
                F += int(j->n_frames);
                B += int(j->n_windows);
            }
            flush();
            // windows: device pointer table
            for (shv_cuda_job* j : jobs) {
                for (size_t e = 0; e < j->n_windows * 20; ++e) {
                    const int32_t wf = j->wframes.empty() ? -1 : j->wframes[e];
                    const int32_t ws = j->wslots.empty() ? -1 : j->wslots[e];
                    const float* p;
                    if (wf >= 0) p = d.d_tokens + size_t(j->frame_off + wf) * TF;
                    else if (ws >= 0) p = d_pool + size_t(ws) * TF;
                    else {
                        std::memcpy(h.tok_stage + size_t(Hn) * TF, j->wtokens[e], TF * 4);
                        p = d.d_tok_stage + size_t(Hn) * TF;
                        ++Hn;
                    }
                    h.tokptr[size_t(j->window_off) * 20 + e] = p;
                }
            }
            if (B) SHV_CUDA_CHECK(cudaMemcpyAsync(d.d_tokptr, h.tokptr, size_t(B) * 20 * sizeof(float*), cudaMemcpyHostToDevice, s_copy));
            if (Hn) SHV_CUDA_CHECK(cudaMemcpyAsync(d.d_tok_stage, h.tok_stage, size_t(Hn) * TF * 4, cudaMemcpyHostToDevice, s_copy));
            if (any_store) SHV_CUDA_CHECK(cudaMemcpyAsync(h.d_store, h.store, size_t(F) * 4, cudaMemcpyHostToDevice, s_copy));
            SHV_CUDA_CHECK(cudaEventRecord(h.h2d, s_copy));
            SHV_CUDA_CHECK(cudaStreamWaitEvent(s_comp, h.h2d, 0));
            SHV_CUDA_CHECK(cudaEventRecord(h.start, s_comp));
            if (F) eng->encode(slot, F, s_comp);
            if (any_store) {
                shv::launch_store_tokens(d.d_tokens, h.d_store, d_pool, F, s_comp);
                SHV_CUDA_CHECK(cudaGetLastError());
            }
            if (B) eng->classify(slot, B, s_comp);
            if (B) SHV_CUDA_CHECK(cudaMemcpyAsync(h.logits, d.d_logits, size_t(B) * 4, cudaMemcpyDeviceToHost, s_comp));
            for (shv_cuda_job* j : jobs) {
                if (!j->tokens_out || !j->n_frames) continue;
                float* dst = j->tokens_out_pinned ? j->tokens_out : h.tokens_out + size_t(j->frame_off) * TF;
                SHV_CUDA_CHECK(cudaMemcpyAsync(dst, d.d_tokens + size_t(j->frame_off) * TF, j->n_frames * TF * 4,
                                               cudaMemcpyDeviceToHost, s_comp));
            }
            SHV_CUDA_CHECK(cudaEventRecord(h.end, s_comp));
            b.F = F;
            b.B = B;
        } catch (const std::exception& e) {
            b.failed = true;
            b.error = e.what();
        }
        return b;
    }

    void finish(Batch& b) {
        SlotHost& h = sh[size_t(b.slot)];
        float gpu_ms = 0.0f;
        if (!b.failed) {
            const cudaError_t e = cudaEventSynchronize(h.end);
            if (e != cudaSuccess) { b.failed = true; b.error = std::string("CUDA error: ") + cudaGetErrorString(e); }
            else if (cudaEventElapsedTime(&gpu_ms, h.start, h.end) != cudaSuccess) { cudaGetLastError(); gpu_ms = 0.0f; }
        } else {
            cudaStreamSynchronize(s_comp);
            cudaStreamSynchronize(s_copy);
        }
        for (shv_cuda_job* j : b.jobs) {
            if (!b.failed) {
                // A non-finite logit (e.g. an fp16 overflow) is reported as NaN = unavailable, never as a score.
                for (size_t i = 0; i < j->n_windows; ++i) {
                    const float v = h.logits[size_t(j->window_off) + i];
                    j->logits_out[i] = std::isfinite(v) ? v : kNaN;
                }
                if (j->tokens_out && !j->tokens_out_pinned && j->n_frames)
                    std::memcpy(j->tokens_out, h.tokens_out + size_t(j->frame_off) * TF, j->n_frames * TF * 4);
            }
            complete(j, !b.failed, b.error, gpu_ms, b.F, b.B);
        }
    }

    void run() {
        if (cudaSetDevice(device) != cudaSuccess) { cudaGetLastError(); }
        std::deque<Batch> inflight;
        int next_slot = 0;
        const int nslots = int(sh.size());
        for (;;) {
            std::vector<shv_cuda_job*> jobs;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (inflight.empty()) cv.wait(lk, [&] { return stopping || !q.empty(); });
                if (q.empty() && inflight.empty() && stopping) break;
                if (!q.empty() && int(inflight.size()) < nslots) {
                    if (opt.max_delay_us > 0 && inflight.empty()) {
                        const auto deadline = Clock::now() + std::chrono::microseconds(opt.max_delay_us);
                        cv.wait_until(lk, deadline, [&] {
                            if (stopping) return true;
                            size_t F = 0, B = 0;
                            for (auto* j : q) { F += j->n_frames; B += j->n_windows; }
                            return F >= size_t(opt.max_frames) || B >= size_t(opt.max_windows);
                        });
                    }
                    take(jobs);
                }
            }
            if (!jobs.empty()) {
                inflight.push_back(launch(next_slot, jobs));
                next_slot = (next_slot + 1) % nslots;
                if (int(inflight.size()) < nslots) {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!q.empty()) continue;  // queue a second batch before waiting on the first
                }
            }
            if (!inflight.empty()) {
                finish(inflight.front());
                inflight.pop_front();
            }
        }
    }
};

}  // namespace

struct shv_cuda_model {
    std::unique_ptr<Core> core;
};

namespace {
template <class F> int guarded(F&& f) {
    try { return f(); } catch (const std::exception& e) { g_err = e.what(); return -1; }
}

// "<weights>.json" sidecar: the export's "output_sha256" must be the SHA-256 of the weight file that is loaded, so a
// threshold (or any other sidecar field) from another export is never paired with these weights.
std::string json_string_field(const std::string& s, const std::string& key) {
    const auto k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    const auto c = s.find(':', k);
    if (c == std::string::npos) return "";
    const auto q0 = s.find('"', c + 1);
    if (q0 == std::string::npos) return "";
    const auto q1 = s.find('"', q0 + 1);
    if (q1 == std::string::npos) return "";
    return s.substr(q0 + 1, q1 - q0 - 1);
}
void check_sidecar_sha(const std::string& weights, const std::string& sha) {
    const std::string path = weights + ".json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string got = json_string_field(ss.str(), "output_sha256");
    if (got.empty()) throw std::runtime_error("no output_sha256 in " + path + " (the sidecar must name its weight file)");
    if (got != sha)
        throw std::runtime_error(path + " output_sha256 " + got + " does not match the weights' SHA-256 " + sha +
                                 " (sidecar of another export)");
}

// "<weights>.cuda.cert" (first line SHIELDV_CUDA_CERT_V2), one line per passing mode:
//   "<weights sha256> <mode> <revision> PASS <cert json> <cert json sha256> threshold=<%.17g>"
// The threshold the certificate was issued for must equal the loaded threshold exactly (double ==), so an edited
// threshold in "<weights>.json" invalidates every certified mode.
std::string check_certificate(const std::string& weights, const std::string& sha, const std::string& mode, double thr) {
    const std::string path = weights + ".cuda.cert";
    std::ifstream in(path);
    if (!in) return "mode " + mode + " needs a quality certificate: " + path + " not found";
    std::string line;
    if (!std::getline(in, line) || line != "SHIELDV_CUDA_CERT_V2")
        return "invalid certificate file " + path + " (expected SHIELDV_CUDA_CERT_V2; V1 lines do not bind the threshold)";
    bool thr_mismatch = false;
    double cert_thr = std::nan("");
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string h, m, rev, verdict, tok;
        if (!(ls >> h >> m >> rev >> verdict)) continue;
        if (!(h == sha && m == mode && rev == SHV_CUDA_REVISION && verdict == "PASS")) continue;
        double t = std::nan("");
        while (ls >> tok)
            if (tok.rfind("threshold=", 0) == 0) {
                const char* b = tok.c_str() + 10;
                char* e = nullptr;
                const double v = std::strtod(b, &e);
                if (e != b && *e == 0) t = v;
            }
        if (std::isfinite(t) && t == thr) return "";
        thr_mismatch = true;
        cert_thr = t;
    }
    if (thr_mismatch) {
        char b[160];
        std::snprintf(b, sizeof b, "certified threshold %.17g, loaded threshold %.17g", cert_thr, thr);
        return "mode " + mode + " certificate in " + path + " does not cover the threshold in " + weights + ".json (" + b +
               ")";
    }
    return "mode " + mode + " has no passing certificate for weights " + sha + " at compute revision " +
           SHV_CUDA_REVISION + " in " + path;
}
}  // namespace

extern "C" {

void shv_cuda_default_options(shv_cuda_options* o) {
    if (!o) return;
    o->max_frames = 640;
    o->max_windows = 512;
    o->max_host_tokens = 2048;
    o->token_slots = 0;
    o->use_graphs = 1;
    o->max_delay_us = 0;
}

shv_cuda_model* shv_cuda_load_ex(const char* weights, const char* mode, int device, const shv_cuda_options* opt) {
    try {
        if (!weights) throw std::runtime_error("null weights path");
        auto core = std::unique_ptr<Core>(new Core());
        core->mode = mode ? mode : "fp32";
        shv::Precision p;
        if (!shv::parse_precision(core->mode, &p)) throw std::runtime_error("unknown mode '" + core->mode + "' (fp32, bf16, fp16)");
        shv_cuda_default_options(&core->opt);
        if (opt) core->opt = *opt;
        const auto& o = core->opt;
        if (o.max_frames < 1 || o.max_frames > 4096 || o.max_windows < 1 || o.max_windows > 4096 || o.max_host_tokens < 0 ||
            o.max_host_tokens > 1 << 20 || o.token_slots < 0 || o.max_delay_us < 0)
            throw std::runtime_error("invalid options");
        core->device = device;
        core->hm = shv::load_host_model(weights);
        core->fingerprint = core->hm.sha256;
        core->threshold = core->hm.threshold;   // load_host_model: present, finite and in (0, 1), else it throws
        check_sidecar_sha(weights, core->fingerprint);
#ifndef SHV_CUDA_EXPERIMENTAL
        if (p != shv::Precision::FP32) {
            const std::string err = check_certificate(weights, core->fingerprint, core->mode, core->threshold);
            if (!err.empty()) throw std::runtime_error(err);
        }
#endif
        core->init();
        auto* m = new shv_cuda_model();
        m->core = std::move(core);
        g_err.clear();
        return m;
    } catch (const std::exception& e) {
        g_err = e.what();
        return nullptr;
    }
}

shv_cuda_model* shv_cuda_load(const char* weights, const char* mode, int device) {
    return shv_cuda_load_ex(weights, mode, device, nullptr);
}

void shv_cuda_free(shv_cuda_model* m) {
    if (!m) return;
    m->core->stop();
    delete m;
}

const char* shv_cuda_last_error(void) { return g_err.c_str(); }
const char* shv_cuda_mode(const shv_cuda_model* m) { return m ? m->core->mode.c_str() : ""; }
const char* shv_cuda_fingerprint(const shv_cuda_model* m) { return m ? m->core->fingerprint.c_str() : ""; }
double shv_cuda_threshold(const shv_cuda_model* m) { return m ? m->core->threshold : std::nan(""); }
const char* shv_cuda_compute_revision(void) { return SHV_CUDA_REVISION; }
int shv_cuda_limits(const shv_cuda_model* m, int* mf, int* mw, int* mh, int* ts) {
    if (!m) { g_err = "null model"; return -1; }
    if (mf) *mf = m->core->opt.max_frames;
    if (mw) *mw = m->core->opt.max_windows;
    if (mh) *mh = m->core->opt.max_host_tokens;
    if (ts) *ts = m->core->opt.token_slots;
    return 0;
}
size_t shv_cuda_weight_bytes(const shv_cuda_model* m) { return m ? m->core->eng->weight_bytes() : 0; }
size_t shv_cuda_device_bytes(const shv_cuda_model* m) { return m ? m->core->device_bytes : 0; }

shv_cuda_job* shv_cuda_submit(shv_cuda_model* m, const shv_cuda_request* r) {
    try {
        if (!m || !r) throw std::runtime_error("null argument");
        Core& c = *m->core;
        const auto& o = c.opt;
        if (r->n_frames > size_t(o.max_frames)) throw std::runtime_error("request has more frames than max_frames");
        if (r->n_windows > size_t(o.max_windows)) throw std::runtime_error("request has more windows than max_windows");
        if (r->n_frames && !r->frames) throw std::runtime_error("null frames");
        if (r->n_windows && !r->logits_out) throw std::runtime_error("null logits_out");
        if (!r->n_frames && !r->n_windows) throw std::runtime_error("empty request");
        auto j = std::unique_ptr<shv_cuda_job>(new shv_cuda_job());
        j->frames = r->frames;
        j->n_frames = r->n_frames;
        j->tokens_out = r->tokens_out;
        j->n_windows = r->n_windows;
        j->logits_out = r->logits_out;
        j->cb = r->callback;
        j->user = r->user;
        if (r->store_slots && r->n_frames) {
            if (!c.d_pool) throw std::runtime_error("store_slots needs options.token_slots > 0");
            j->store_slots.assign(r->store_slots, r->store_slots + r->n_frames);
            std::unordered_set<int32_t> seen;
            for (int32_t s : j->store_slots) {
                if (s < -1 || s >= o.token_slots) throw std::runtime_error("store slot out of range");
                if (s >= 0 && !seen.insert(s).second) throw std::runtime_error("duplicate store slot in one request");
            }
        }
        const size_t E = r->n_windows * 20;
        if (E) {
            if (r->window_frames) j->wframes.assign(r->window_frames, r->window_frames + E);
            if (r->window_slots) j->wslots.assign(r->window_slots, r->window_slots + E);
            if (r->window_tokens) j->wtokens.assign(r->window_tokens, r->window_tokens + E);
            for (size_t e = 0; e < E; ++e) {
                const int32_t wf = j->wframes.empty() ? -1 : j->wframes[e];
                const int32_t ws = j->wslots.empty() ? -1 : j->wslots[e];
                if (wf >= 0) {
                    if (size_t(wf) >= r->n_frames) throw std::runtime_error("window frame index out of range");
                } else if (ws >= 0) {
                    if (!c.d_pool || ws >= o.token_slots) throw std::runtime_error("window slot out of range");
                } else {
                    if (j->wtokens.empty() || !j->wtokens[e]) throw std::runtime_error("window entry has no token source");
                    ++j->host_tokens;
                }
            }
            if (j->host_tokens > o.max_host_tokens) throw std::runtime_error("request has more host token blocks than max_host_tokens");
        }
        j->frames_pinned = is_pinned(r->frames, r->n_frames * FB);
        j->tokens_out_pinned = r->tokens_out && is_pinned(r->tokens_out, r->n_frames * TF * 4);
        j->t_submit = Clock::now();
        shv_cuda_job* raw = j.release();
        {
            std::lock_guard<std::mutex> lk(c.mu);
            if (c.stopping) { delete raw; throw std::runtime_error("model is stopping"); }
            c.q.push_back(raw);
        }
        c.cv.notify_all();
        g_err.clear();
        return raw;
    } catch (const std::exception& e) {
        g_err = e.what();
        return nullptr;
    }
}

int shv_cuda_job_status(const shv_cuda_job* j) {
    if (!j) return -1;
    std::lock_guard<std::mutex> lk(j->m);
    return j->state;
}
int shv_cuda_job_wait(shv_cuda_job* j) {
    if (!j) { g_err = "null job"; return -1; }
    std::unique_lock<std::mutex> lk(j->m);
    j->cv.wait(lk, [&] { return j->state != 0; });
    if (j->state < 0) { g_err = j->error; return -1; }
    return 0;
}
const char* shv_cuda_job_error(const shv_cuda_job* j) {
    if (!j) return "null job";
    std::lock_guard<std::mutex> lk(j->m);
    return j->error.c_str();
}
int shv_cuda_job_timing(const shv_cuda_job* j, double* queue_ms, double* run_ms, double* gpu_ms, int* bf, int* bw) {
    if (!j) return -1;
    std::lock_guard<std::mutex> lk(j->m);
    if (j->state == 0) return 1;
    auto ms = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    if (queue_ms) *queue_ms = ms(j->t_start - j->t_submit);
    if (run_ms) *run_ms = ms(j->t_done - j->t_start);
    if (gpu_ms) *gpu_ms = j->gpu_ms;
    if (bf) *bf = j->batch_frames;
    if (bw) *bw = j->batch_windows;
    return 0;
}
void shv_cuda_job_free(shv_cuda_job* j) {
    if (!j) return;
    {
        std::unique_lock<std::mutex> lk(j->m);
        j->cv.wait(lk, [&] { return j->released; });
    }
    delete j;
}

void* shv_cuda_host_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        g_err = "cudaHostAlloc failed";
        return nullptr;
    }
    return p;
}
void shv_cuda_host_free(void* p) {
    if (p) cudaFreeHost(p);
}

// ---------------------------------------------------------------- synchronous wrappers
static int wait_all(std::vector<shv_cuda_job*>& jobs) {
    int rc = 0;
    std::string err;
    for (auto* j : jobs) {
        if (shv_cuda_job_wait(j) != 0) { rc = -1; if (err.empty()) err = shv_cuda_job_error(j); }
        shv_cuda_job_free(j);
    }
    if (rc) g_err = err;
    return rc;
}

int shv_cuda_encode_frames(shv_cuda_model* m, const uint8_t* rgb, size_t n, float* tokens) {
    return guarded([&] {
        if (!m || (n && (!rgb || !tokens))) throw std::runtime_error("null argument");
        const size_t step = size_t(m->core->opt.max_frames);
        std::vector<shv_cuda_job*> jobs;
        for (size_t off = 0; off < n; off += step) {
            shv_cuda_request r{};
            r.frames = rgb + off * FB;
            r.n_frames = std::min(step, n - off);
            r.tokens_out = tokens + off * TF;
            shv_cuda_job* j = shv_cuda_submit(m, &r);
            if (!j) { const std::string e = g_err; wait_all(jobs); std::fill(tokens, tokens + n * TF, kNaN); throw std::runtime_error(e); }
            jobs.push_back(j);
        }
        return wait_all(jobs);
    });
}

int shv_cuda_classify(shv_cuda_model* m, const float* tokens, size_t b, float* logits) {
    return guarded([&] {
        if (!m || (b && (!tokens || !logits))) throw std::runtime_error("null argument");
        const auto& o = m->core->opt;
        const size_t step = size_t(std::max(1, std::min(o.max_windows, o.max_host_tokens / 20)));
        if (o.max_host_tokens < 20) throw std::runtime_error("classify needs max_host_tokens >= 20");
        std::vector<shv_cuda_job*> jobs;
        std::vector<const float*> ptrs;
        for (size_t off = 0; off < b; off += step) {
            const size_t w = std::min(step, b - off);
            ptrs.resize(w * 20);
            for (size_t e = 0; e < w * 20; ++e) ptrs[e] = tokens + (off * 20 + e) * TF;
            shv_cuda_request r{};
            r.n_windows = w;
            r.window_tokens = ptrs.data();
            r.logits_out = logits + off;
            shv_cuda_job* j = shv_cuda_submit(m, &r);
            if (!j) { const std::string e = g_err; wait_all(jobs); std::fill(logits, logits + b, kNaN); throw std::runtime_error(e); }
            jobs.push_back(j);
        }
        return wait_all(jobs);
    });
}

int shv_cuda_predict(shv_cuda_model* m, const uint8_t* clips, size_t b, float* logits) {
    return guarded([&] {
        if (!m || (b && (!clips || !logits))) throw std::runtime_error("null argument");
        const auto& o = m->core->opt;
        const size_t step = size_t(std::min(o.max_windows, o.max_frames / 20));
        if (step < 1) throw std::runtime_error("predict needs max_frames >= 20");
        std::vector<int32_t> idx(step * 20);
        for (size_t e = 0; e < idx.size(); ++e) idx[e] = int32_t(e);
        std::vector<shv_cuda_job*> jobs;
        for (size_t off = 0; off < b; off += step) {
            const size_t w = std::min(step, b - off);
            shv_cuda_request r{};
            r.frames = clips + off * 20 * FB;
            r.n_frames = w * 20;
            r.n_windows = w;
            r.window_frames = idx.data();
            r.logits_out = logits + off;
            shv_cuda_job* j = shv_cuda_submit(m, &r);
            if (!j) { const std::string e = g_err; wait_all(jobs); std::fill(logits, logits + b, kNaN); throw std::runtime_error(e); }
            jobs.push_back(j);
        }
        return wait_all(jobs);
    });
}

}  // extern "C"
