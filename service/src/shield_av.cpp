// SHIELD-AV live service core: per-participant audio+video sessions over the native SHIELD-A audio engine
// (libshield.so, shield_stream), the SHIELD-V preprocessing library (libshield_v_prep.so, shv_participant) and the
// SHIELD-V model libraries (libshield_v_cuda.so async jobs + device token slots, or libshield_v_cpu.so workers).
// See service/include/shield_av.h for the contract.
#include "shield_av.h"

#include "shield.h"
#include "shv_cpu.h"
#include "shv_cuda.h"
#include "shv_prep.h"

#include <dlfcn.h>
#include <pthread.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

thread_local std::string g_err;
constexpr size_t FB = 180 * 320 * 3;      // one crop
constexpr size_t TF = 16 * 384;           // tokens of one frame
constexpr int NSLOT = 24;                 // token cache entries per session
constexpr int HOP = SHAV_HOP_SAMPLES;
const float kNaNf = std::numeric_limits<float>::quiet_NaN();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

double mono() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
}
int fail(const std::string& m) {
    g_err = m;
    return -1;
}
std::string sdup(const char* p) { return p ? std::string(p) : std::string(); }

// Threads created inside a library inherit the creating thread's name; name them by component for CPU accounting.
struct ThreadName {
    char old[32] = {0};
    explicit ThreadName(const char* n) {
        pthread_getname_np(pthread_self(), old, sizeof old);
        pthread_setname_np(pthread_self(), n);
    }
    ~ThreadName() { pthread_setname_np(pthread_self(), old); }
};



struct AudioJob {
    shav_session* s = nullptr;
    int64_t hop = 0;
    std::vector<float> pcm;
    double t_input = 0, t_enq = 0;
};

struct VideoJob {
    shav_session* s = nullptr;
    int64_t hop = 0, window = 0;
    int n_new = 0, n_hit = 0, anchors = 0;
    uint8_t* crops = nullptr;                 // n_new crops (pool buffer of 20)
    int entry[20];                            // token-cache entry of each window position
    int32_t store[20], wframes[20], wslots[20];
    bool has_hits = false;
    int64_t gen = 0;                          // token-cache generation at planning time
    float logit = kNaNf;
    int status = 0;                           // 1 done, -1 failed
    shv_cuda_job* cj = nullptr;
    double t_input = 0, t_submit = 0;
    float queue_ms = kNaNf, compute_ms = kNaNf;
    int batch_windows = 0;
};

struct CacheEntry {
    bool valid = false;
    int64_t k = -1;
    shv_rect r{0, 0, 0, 0};
    uint64_t stamp = 0;
};

struct HopState {
    bool a_final = false, v_final = false, v_resolving = false;
    shav_modality_result a, v;
};

void init_part(shav_modality_result& r, double thr, double ws, double we, bool audio) {
    std::memset(&r, 0, sizeof r);
    r.status = SHAV_COLLECTING;
    r.reason = SHAV_R_WARMUP;
    r.logit = kNaNf;
    r.score = kNaN;
    r.decision = -1;
    r.alert = -1;
    r.threshold = thr;
    r.window_start = ws;
    r.window_end = we;
    r.valid_speech_sec = audio ? 0.0 : kNaN;
    r.face_detection_anchors = -1;
    r.frames_encoded = -1;
    r.frames_reused = -1;
    r.t_input = kNaN;
    r.t_done = kNaN;
    r.queue_ms = kNaNf;
    r.compute_ms = kNaNf;
    r.batch_windows = 0;
}

// Driver API (libcuda.so.1, loaded at run time): set CU_CTX_SCHED_BLOCKING_SYNC on the device's primary context
// before the CUDA library's statically linked runtime retains it.  The ordinal honours CUDA_VISIBLE_DEVICES.
bool cuda_blocking_sync(int device, std::string& log) {
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        log += "blocking_sync: libcuda.so.1 not found; ";
        return false;
    }
    typedef int (*init_t)(unsigned);
    typedef int (*devget_t)(int*, int);
    typedef int (*flags_t)(int, unsigned);
    typedef int (*getstate_t)(int, unsigned*, int*);
    auto cu_init = reinterpret_cast<init_t>(dlsym(h, "cuInit"));
    auto cu_dev = reinterpret_cast<devget_t>(dlsym(h, "cuDeviceGet"));
    auto cu_set = reinterpret_cast<flags_t>(dlsym(h, "cuDevicePrimaryCtxSetFlags_v2"));
    if (!cu_set) cu_set = reinterpret_cast<flags_t>(dlsym(h, "cuDevicePrimaryCtxSetFlags"));
    auto cu_get = reinterpret_cast<getstate_t>(dlsym(h, "cuDevicePrimaryCtxGetState"));
    int dev = 0, rc = -1;
    if (cu_init && cu_dev && cu_set && cu_init(0) == 0 && cu_dev(&dev, device) == 0) rc = cu_set(dev, 0x04u);
    unsigned flags = 0;
    int active = 0;
    if (rc == 0 && cu_get && cu_get(dev, &flags, &active) == 0 && (flags & 0x07u) == 0x04u) {
        log += "blocking_sync on; ";
        return true;   // keep libcuda loaded (the runtime uses it anyway)
    }
    log += "blocking_sync failed (driver error " + std::to_string(rc) + "); ";
    return false;
}

}  // namespace

// ================================================================================================= engine
struct shav_engine {
    shav_engine_config cfg{};
    std::string version, audio_weights, video_weights, audio_mode_s, video_backend_s, video_mode_s, audio_fp, audio_rev,
        video_fp, video_rev, load_log, cpu_isa_s;
    shield_model* amodel = nullptr;
    int backend = 0;   // 0 none, 1 cuda, 2 cpu
    shv_cuda_model* cuda = nullptr;
    shv_cpu_model* cpu = nullptr;
    shv_cascade* cascade = nullptr;
    double vthr = kNaN;
    size_t device_bytes = 0;

    std::mutex slot_mu;
    std::vector<int> free_blocks;

    std::mutex pool_mu;
    std::vector<uint8_t*> pool;
    std::vector<uint8_t*> pool_all;

    std::mutex aq_mu;
    std::condition_variable aq_cv;
    std::deque<AudioJob*> aq;
    std::vector<std::thread> adrv;

    std::mutex vq_mu;
    std::condition_variable vq_cv;
    std::deque<VideoJob*> vq;     // CUDA completions
    std::thread vdone;

    std::mutex cq_mu;
    std::condition_variable cq_cv;
    std::deque<VideoJob*> cq;     // CPU work
    std::vector<std::thread> cw;

    bool stopping = false;
    std::atomic<int64_t> n_sessions{0}, n_audio{0}, n_video{0}, n_enc{0}, n_reuse{0}, n_drop{0};

    uint8_t* get_buf() {
        std::lock_guard<std::mutex> lk(pool_mu);
        if (!pool.empty()) {
            uint8_t* b = pool.back();
            pool.pop_back();
            return b;
        }
        uint8_t* b = backend == 1 ? static_cast<uint8_t*>(shv_cuda_host_alloc(20 * FB))
                                  : static_cast<uint8_t*>(std::aligned_alloc(64, 20 * FB));
        if (b) pool_all.push_back(b);
        return b;
    }
    void put_buf(uint8_t* b) {
        if (!b) return;
        std::lock_guard<std::mutex> lk(pool_mu);
        pool.push_back(b);
    }
    // grow the pool by one staging buffer (pinned on CUDA) outside the push path
    void reserve_buf() {
        uint8_t* b = backend == 1 ? static_cast<uint8_t*>(shv_cuda_host_alloc(20 * FB))
                                  : static_cast<uint8_t*>(std::aligned_alloc(64, 20 * FB));
        if (!b) return;
        std::lock_guard<std::mutex> lk(pool_mu);
        pool_all.push_back(b);
        pool.push_back(b);
    }
    void audio_loop();
    void video_done_loop();
    void cpu_loop();
    void submit_video(VideoJob* j);
};

// ================================================================================================= session
struct shav_session {
    shav_engine* e = nullptr;
    std::string id;
    shav_session_config cfg{};
    bool audio_on = false, video_on = false;
    double audio_wait = 2.0;   // resolved cfg.audio_wait_s (NaN = video_wait_s); +inf disables the audio lag rule
    std::atomic<bool> audio_started{false};   // any audio sample pushed since create / reset

    // ---- hop states / emission (mu)
    std::mutex mu;
    std::map<int64_t, HopState> hops;
    int64_t next_emit = 0, v_settle_from = 0, audio_clock = 0;   // audio_clock = audio hops with complete input
    std::deque<shav_hop_result> out;
    std::deque<int> v_hist;
    bool finished = false, discard = false;
    int64_t emitted = 0;

    // ---- audio input (amu)
    std::mutex amu;
    shield_stream* stream = nullptr;
    std::vector<float> hopbuf;
    size_t fill = 0;
    int64_t audio_hops = 0;
    bool a_busy = false;
    std::deque<AudioJob*> a_pending;
    std::atomic<size_t> stream_bytes{0};

    // ---- video input (vmu)
    std::mutex vmu;
    shv_participant* part = nullptr;
    double last_t = -std::numeric_limits<double>::infinity();
    double t0 = 0, vstart = 0;
    int64_t next_window = 0;
    CacheEntry cache[NSLOT];
    uint64_t stamp = 0;
    int slot_base = -1;
    float* host_tok = nullptr;
    std::vector<uint8_t> tapbuf;
    std::atomic<bool> video_started{false};
    std::atomic<int64_t> first_hop{-1};
    // a failed job poisons its cache generation: pending jobs of that generation that read cached tokens fail too,
    // and the next window plan clears the cache and starts a new generation
    std::atomic<int64_t> cache_gen{0}, poison_gen{-1};

    // ---- video jobs (jmu)
    std::mutex jmu;
    bool v_busy = false;
    std::deque<VideoJob*> v_pending;
    std::atomic<int> crops_held{0};

    // ---- idle tracking
    std::mutex imu;
    std::condition_variable icv;
    int inflight = 0;

    // ---- stats
    std::atomic<int64_t> st_vres{0}, st_vscored{0}, st_pushed{0}, st_sampled{0}, st_enc{0}, st_reuse{0};

    double hop_ws(int64_t h) const { return 2.0 * double(h) - 8.0; }
    void job_add() {
        std::lock_guard<std::mutex> lk(imu);
        ++inflight;
    }
    void job_done() {
        std::lock_guard<std::mutex> lk(imu);
        --inflight;
        icv.notify_all();
    }

    // ---- under mu
    HopState& hop_state(int64_t h) {
        auto it = hops.find(h);
        if (it != hops.end()) return it->second;
        HopState& hs = hops[h];
        init_part(hs.a, cfg.audio_threshold, hop_ws(h), hop_ws(h) + 10.0, true);
        init_part(hs.v, e->vthr, hop_ws(h), hop_ws(h) + 10.0, false);
        if (!audio_on) {
            hs.a.status = SHAV_UNAVAILABLE;
            hs.a.reason = SHAV_R_DISABLED;
            hs.a.valid_speech_sec = kNaN;
            hs.a.t_done = mono();
            hs.a_final = true;
        }
        if (!video_on) {
            hs.v.status = SHAV_UNAVAILABLE;
            hs.v.reason = SHAV_R_DISABLED;
            hs.v.t_done = mono();
            hs.v_final = true;
        } else if (video_started.load() && h < first_hop.load()) {
            hs.v.reason = SHAV_R_WARMUP;
            hs.v.t_done = mono();
            hs.v_final = true;
        }
        return hs;
    }
    void video_final(HopState& hs, int status, int reason) {
        hs.v.status = status;
        hs.v.reason = reason;
        hs.v.t_done = mono();
        hs.v_final = true;
    }
    // audio clock reached hop H's end (H complete): settle video parts that can no longer change
    void settle_video(int64_t H, bool ending) {
        if (!video_on) return;
        const bool started = video_started.load();
        const int64_t fh = first_hop.load();
        for (int64_t h = std::max(v_settle_from, next_emit); h <= H; ++h) {
            HopState& hs = hop_state(h);
            if (hs.v_final || hs.v_resolving) continue;
            if (!started)
                video_final(hs, SHAV_COLLECTING, SHAV_R_NO_INPUT);
            else if (h < fh)
                video_final(hs, SHAV_COLLECTING, SHAV_R_WARMUP);
            else if (ending)
                video_final(hs, SHAV_UNAVAILABLE, SHAV_R_ENDED);
            else if (2.0 * double(H - h) >= cfg.video_wait_s - 1e-9)
                video_final(hs, SHAV_UNAVAILABLE, SHAV_R_LAG);
        }
        while (true) {
            auto it = hops.find(v_settle_from);
            if (v_settle_from < next_emit || (it != hops.end() && it->second.v_final)) ++v_settle_from;
            else break;
        }
    }
    // Audio lag rule, the mirror of settle_video: the video clock reached hop H's end + audio_wait (or the input
    // ended) and the audio input of hops audio_clock..H has not arrived (microphone muted / dropped / stalled).  Those
    // audio parts become final now so the hops, with their video parts, are released in order.  Hops whose audio
    // input is complete (h < audio_clock) are being computed and are never forced.  Audio that arrives later for a
    // forced hop is still run through the audio stream (its ring needs every hop) but its result is discarded.
    void settle_audio(int64_t H, bool ending) {
        if (!audio_on) return;
        const bool started = audio_started.load();
        for (int64_t h = std::max(audio_clock, next_emit); h <= H; ++h) {
            HopState& hs = hop_state(h);
            if (hs.a_final) continue;
            shav_modality_result& a = hs.a;
            if (!started) {
                a.status = SHAV_COLLECTING;
                a.reason = SHAV_R_NO_INPUT;
            } else if (h < 4) {
                a.status = SHAV_COLLECTING;   // the audio engine reports hops 0..3 as collecting in any case
                a.reason = SHAV_R_WARMUP;
            } else {
                a.status = SHAV_UNAVAILABLE;
                a.reason = ending ? SHAV_R_ENDED : SHAV_R_LAG;
            }
            a.valid_speech_sec = kNaN;
            a.t_done = mono();
            hs.a_final = true;
        }
    }
    // The video clock moved (under vmu + mu): tv = session seconds of the newest pushed frame, h_resolved = the last
    // hop whose video window has resolved.  Hop h is overdue for audio when 2(h+1) + audio_wait <= tv; only hops up
    // to h_resolved are forced (their video part exists; it also bounds the loop for a timestamp jump).
    void video_clock(double tv, int64_t h_resolved) {
        if (!audio_on || !std::isfinite(audio_wait)) return;
        double x = (tv - audio_wait) / 2.0 + 1e-9;
        if (!(x >= 1.0)) return;
        if (x > 9e15) x = 9e15;
        settle_audio(std::min(int64_t(std::floor(x)) - 1, h_resolved), false);
    }
    // shav_session_finish (under mu): input ended.  Unresolved video parts up to the last hop either modality reached
    // become ended (video), and so do the missing audio parts of hops whose video part exists (audio), so no result
    // that was computed is held back.  Audio-disabled sessions are clocked by the video and need nothing here.
    void settle_end() {
        if (!audio_on) return;
        int64_t H = audio_clock - 1;
        if (video_on && !hops.empty()) H = std::max(H, hops.rbegin()->first);
        settle_audio(H, true);
        settle_video(H, true);
    }
    int combined(const HopState& hs) const {
        const bool av = hs.a.status == SHAV_VALID, vv = hs.v.status == SHAV_VALID;
        switch (cfg.policy) {
            case SHAV_POLICY_AUDIO: return av ? hs.a.alert : -1;
            case SHAV_POLICY_VIDEO: return vv ? hs.v.alert : -1;
            case SHAV_POLICY_BOTH:
                if (av && vv) return (hs.a.alert == 1 && hs.v.alert == 1) ? 1 : 0;
                if ((av && hs.a.alert == 0) || (vv && hs.v.alert == 0)) return 0;
                return -1;
            default:
                if ((av && hs.a.alert == 1) || (vv && hs.v.alert == 1)) return 1;
                if (av || vv) return 0;
                return -1;
        }
    }
    void try_emit() {
        for (;;) {
            auto it = hops.find(next_emit);
            if (it == hops.end() || !it->second.a_final || !it->second.v_final) break;
            shav_hop_result r;
            std::memset(&r, 0, sizeof r);
            r.hop = next_emit;
            r.hop_end = 2.0 * double(next_emit + 1);
            r.policy = cfg.policy;
            r.audio = it->second.a;
            r.video = it->second.v;
            r.alert = combined(it->second);
            r.t_emit = mono();
            hops.erase(it);
            ++next_emit;
            ++emitted;
            if (cfg.callback)
                cfg.callback(&r, cfg.user);
            else
                out.push_back(r);
        }
    }
};

namespace {

// ------------------------------------------------------------------------------------------------- audio
void enqueue_audio(shav_engine* e, AudioJob* j) {
    j->t_enq = mono();
    {
        std::lock_guard<std::mutex> lk(e->aq_mu);
        e->aq.push_back(j);
    }
    e->aq_cv.notify_one();
}

void deliver_audio(shav_session* s, AudioJob* j, const shield_result* r, bool ok, double qms, double cms) {
    shav_modality_result m;
    init_part(m, s->cfg.audio_threshold, s->hop_ws(j->hop), s->hop_ws(j->hop) + 10.0, true);
    m.t_input = j->t_input;
    m.queue_ms = float(qms);
    m.compute_ms = float(cms);
    if (!ok) {
        m.status = SHAV_UNAVAILABLE;
        m.reason = SHAV_R_FAILED;
    } else {
        m.window_start = r->window_start;
        m.window_end = r->window_end;
        m.valid_speech_sec = r->valid_speech_sec;
        switch (r->status) {
            case SHIELD_COLLECTING: m.status = SHAV_COLLECTING; m.reason = SHAV_R_WARMUP; break;
            case SHIELD_INSUFFICIENT_SPEECH: m.status = SHAV_INSUFFICIENT_SPEECH; m.reason = SHAV_R_SILENCE; break;
            case SHIELD_VALID:
                if (std::isfinite(r->logit)) {
                    m.status = SHAV_VALID;
                    m.reason = SHAV_R_NONE;
                    m.logit = r->logit;
                    m.score = r->score;
                    m.decision = r->score >= s->cfg.audio_threshold ? 1 : 0;
                    m.alert = r->alert;
                } else {
                    m.status = SHAV_UNAVAILABLE;
                    m.reason = SHAV_R_FAILED;
                }
                break;
            default:
                m.status = SHAV_UNAVAILABLE;
                m.reason = s->e->amodel ? SHAV_R_FAILED : SHAV_R_DISABLED;
        }
    }
    m.t_done = mono();
    std::lock_guard<std::mutex> lk(s->mu);
    if (s->discard) return;
    if (j->hop < s->next_emit) return;   // forced by the audio lag rule and already emitted
    HopState& hs = s->hop_state(j->hop);
    if (hs.a_final) return;              // forced by the audio lag rule (not yet emitted)
    hs.a = m;
    hs.a_final = true;
    s->try_emit();
}

}  // namespace

void shav_engine::audio_loop() {
    for (;;) {
        AudioJob* j;
        {
            std::unique_lock<std::mutex> lk(aq_mu);
            aq_cv.wait(lk, [&] { return stopping || !aq.empty(); });
            if (aq.empty()) return;
            j = aq.front();
            aq.pop_front();
        }
        shav_session* s = j->s;
        const double t_start = mono();
        shield_result r{};
        bool ok = false;
        double q = 0, c = 0, done = 0;
        shield_ticket* t = shield_stream_submit_hop(s->stream, j->pcm.data(), 2.0 * double(j->hop + 1));
        if (t) {
            ok = shield_ticket_wait(t, &r) == 0;
            if (shield_ticket_timing(t, &q, &c, &done) != 0) q = c = 0;
            shield_ticket_free(t);
        }
        n_audio.fetch_add(1);
        s->stream_bytes.store(shield_stream_bytes(s->stream));   // not busy now: never blocks
        deliver_audio(s, j, &r, ok, (t_start - j->t_enq) * 1e3 + q, c);
        AudioJob* next = nullptr;
        {
            std::lock_guard<std::mutex> lk(s->amu);
            if (!s->a_pending.empty()) {
                next = s->a_pending.front();
                s->a_pending.pop_front();
            } else {
                s->a_busy = false;
            }
        }
        delete j;
        if (next) enqueue_audio(this, next);
        s->job_done();
    }
}

// ------------------------------------------------------------------------------------------------- video
namespace {

// Runs on the CUDA library's worker thread.  The job handle is taken from the callback argument: the submitting thread
// must not touch the VideoJob after shv_cuda_submit succeeded, because the job can complete (and finish_video can
// delete the VideoJob on the shav-vdone thread) before shv_cuda_submit even returns.
void cuda_cb(shv_cuda_job* cj, int status, void* user) {
    VideoJob* j = static_cast<VideoJob*>(user);
    j->cj = cj;
    j->status = status;
    shav_engine* e = j->s->e;
    {
        std::lock_guard<std::mutex> lk(e->vq_mu);
        e->vq.push_back(j);
    }
    e->vq_cv.notify_one();
}

void finish_video(shav_engine* e, VideoJob* j);

}  // namespace

void shav_engine::submit_video(VideoJob* j) {
    j->t_submit = mono();
    if (backend == 1) {
        shv_cuda_request r{};
        r.frames = j->crops;
        r.n_frames = size_t(j->n_new);
        r.store_slots = j->store;
        r.n_windows = 1;
        r.window_frames = j->wframes;
        r.window_slots = j->wslots;
        r.logits_out = &j->logit;
        r.callback = cuda_cb;
        r.user = j;
        // On success, ownership of j passes to the completion path (cuda_cb -> video_done_loop): do not read or write
        // *j here afterwards.  On failure no callback runs and j is still ours.
        if (!shv_cuda_submit(cuda, &r)) {
            j->cj = nullptr;
            j->status = -1;
            std::lock_guard<std::mutex> lk(vq_mu);
            vq.push_back(j);
            vq_cv.notify_one();
        }
    } else {
        {
            std::lock_guard<std::mutex> lk(cq_mu);
            cq.push_back(j);
        }
        cq_cv.notify_one();
    }
}

void shav_engine::video_done_loop() {
    for (;;) {
        VideoJob* j;
        {
            std::unique_lock<std::mutex> lk(vq_mu);
            vq_cv.wait(lk, [&] { return stopping || !vq.empty(); });
            if (vq.empty()) return;
            j = vq.front();
            vq.pop_front();
        }
        if (j->cj) {
            double q = 0, r = 0, g = 0;
            int bf = 0, bw = 0;
            if (shv_cuda_job_timing(j->cj, &q, &r, &g, &bf, &bw) == 0) {
                j->queue_ms = float(q);
                j->compute_ms = float(r);
                j->batch_windows = bw;
            }
            if (shv_cuda_job_status(j->cj) != 1) j->status = -1;
            shv_cuda_job_free(j->cj);
            j->cj = nullptr;
        }
        finish_video(this, j);
    }
}

void shav_engine::cpu_loop() {
    std::vector<const uint8_t*> fr(20);
    std::vector<float*> tk(20);
    std::vector<const float*> wt(20);
    for (;;) {
        VideoJob* j;
        {
            std::unique_lock<std::mutex> lk(cq_mu);
            cq_cv.wait(lk, [&] { return stopping || !cq.empty(); });
            if (cq.empty()) return;
            j = cq.front();
            cq.pop_front();
        }
        const double t0 = mono();
        j->queue_ms = float((t0 - j->t_submit) * 1e3);
        shav_session* s = j->s;
        int n = 0;
        for (int i = 0; i < 20; ++i) {
            if (j->wframes[i] >= 0) {
                fr[size_t(n)] = j->crops + size_t(j->wframes[i]) * FB;
                tk[size_t(n)] = s->host_tok + size_t(j->entry[i]) * TF;
                ++n;
            }
            wt[size_t(i)] = s->host_tok + size_t(j->entry[i]) * TF;
        }
        int rc = shv_cpu_encode_frame_ptrs(cpu, fr.data(), size_t(n), tk.data());
        if (rc == 0) rc = shv_cpu_classify_ptrs(cpu, wt.data(), 1, &j->logit);
        j->status = rc == 0 ? 1 : -1;
        j->compute_ms = float((mono() - t0) * 1e3);
        finish_video(this, j);
    }
}

namespace {

void deliver_video_failed(shav_session* s, VideoJob* j, int reason) {
    std::lock_guard<std::mutex> lk(s->mu);
    if (s->discard) return;
    HopState& hs = s->hop_state(j->hop);
    hs.v_resolving = false;
    if (hs.v_final) return;
    hs.v.t_input = j->t_input;
    hs.v.status = SHAV_UNAVAILABLE;
    hs.v.reason = reason;
    hs.v.t_done = mono();
    hs.v_final = true;
    s->try_emit();
}

void finish_video(shav_engine* e, VideoJob* j) {
    shav_session* s = j->s;
    e->n_video.fetch_add(1);
    const bool ok = j->status == 1 && std::isfinite(j->logit);
    if (!ok) {
        for (int64_t g = s->poison_gen.load(); g < j->gen && !s->poison_gen.compare_exchange_weak(g, j->gen);) {
        }
        deliver_video_failed(s, j, SHAV_R_FAILED);
    } else {
        std::lock_guard<std::mutex> lk(s->mu);
        if (!s->discard) {
            HopState& hs = s->hop_state(j->hop);
            hs.v_resolving = false;
            if (!hs.v_final) {
                shav_modality_result& m = hs.v;
                m.status = SHAV_VALID;
                m.reason = SHAV_R_NONE;
                m.logit = j->logit;
                m.score = 1.0 / (1.0 + std::exp(-double(j->logit)));
                m.decision = m.score >= e->vthr ? 1 : 0;
                s->v_hist.push_back(m.decision);
                while (int(s->v_hist.size()) > s->cfg.video_alert_n) s->v_hist.pop_front();
                int c = 0;
                for (int d : s->v_hist) c += d;
                m.alert = c >= s->cfg.video_alert_k ? 1 : 0;
                m.face_detection_anchors = j->anchors;
                m.frames_encoded = j->n_new;
                m.frames_reused = j->n_hit;
                m.t_input = j->t_input;
                m.queue_ms = j->queue_ms;
                m.compute_ms = j->compute_ms;
                m.batch_windows = j->batch_windows;
                m.t_done = mono();
                hs.v_final = true;
                s->st_vscored.fetch_add(1);
                s->try_emit();
            }
        }
    }
    e->put_buf(j->crops);
    s->crops_held.fetch_sub(1);
    delete j;
    // backpressure: next pending window of this session
    for (;;) {
        VideoJob* next = nullptr;
        {
            std::lock_guard<std::mutex> lk(s->jmu);
            if (s->v_pending.empty()) {
                s->v_busy = false;
                break;
            }
            next = s->v_pending.front();
            s->v_pending.pop_front();
        }
        if (next->has_hits && next->gen <= s->poison_gen.load()) {
            // its cached tokens may come from a failed (or dropped) job of the same cache generation
            deliver_video_failed(s, next, SHAV_R_FAILED);
            e->put_buf(next->crops);
            s->crops_held.fetch_sub(1);
            delete next;
            s->job_done();
            continue;
        }
        e->submit_video(next);
        break;
    }
    s->job_done();
}

}  // namespace

// ================================================================================================= C API
extern "C" {

const char* shav_last_error(void) { return g_err.c_str(); }
const char* shav_version(void) { return "shield-av 1.1 (api 2)"; }
double shav_monotonic(void) { return mono(); }

const char* shav_status_name(int s) {
    switch (s) {
        case SHAV_COLLECTING: return "collecting";
        case SHAV_VALID: return "valid";
        case SHAV_UNAVAILABLE: return "unavailable";
        case SHAV_INSUFFICIENT_SPEECH: return "insufficient_speech";
        default: return "?";
    }
}
const char* shav_reason_name(int r) {
    static const char* n[] = {"none", "warmup", "no_input", "gap", "lag", "disabled", "failed", "overload", "silence", "ended"};
    return r >= 0 && r <= 9 ? n[r] : "?";
}

void shav_engine_default_config(shav_engine_config* c) {
    if (!c) return;
    std::memset(c, 0, sizeof *c);
    c->audio_mode = nullptr;
    c->audio_workers = 4;
    c->audio_cache_level = -1;
    c->video_backend = "auto";
    c->cuda_device = 0;
    c->cuda_max_delay_us = 0;
    c->cuda_max_frames = 640;
    c->cuda_max_windows = 512;
    c->max_sessions = 512;
    c->cpu_video_workers = 4;
    c->cpu_video_threads = 1;
    c->version = "shield-av-1";
}

void shav_session_default_config(shav_session_config* c) {
    if (!c) return;
    std::memset(c, 0, sizeof *c);
    c->audio_enabled = 1;
    c->video_enabled = 1;
    c->audio_threshold = 0.5;
    c->audio_alert_k = 1;
    c->audio_alert_n = 1;
    c->audio_lowpass = 1;
    c->video_alert_k = 1;
    c->video_alert_n = 1;
    c->policy = SHAV_POLICY_EITHER;
    c->video_start = kNaN;
    c->video_first_hop = 4;
    c->video_wait_s = 2.0;
    c->video_cache = 1;
    c->video_store = SHV_STORE_AUTO;
    c->video_max_pending = 2;
    c->audio_wait_s = kNaN;   // = video_wait_s
}

shav_engine* shav_engine_create(const shav_engine_config* cfg_in) {
    shav_engine_config dc;
    shav_engine_default_config(&dc);
    const shav_engine_config& c = cfg_in ? *cfg_in : dc;
    if (c.audio_workers < 1 || c.audio_workers > 8) return (fail("audio_workers must be 1..8"), nullptr);
    if (c.max_sessions < 1 || c.max_sessions > 100000) return (fail("max_sessions must be 1..100000"), nullptr);
    if (c.cpu_video_workers < 1 || c.cpu_video_workers > 64 || c.cpu_video_threads < 1 || c.cpu_video_threads > 64)
        return (fail("cpu_video_workers / cpu_video_threads must be 1..64"), nullptr);
    std::unique_ptr<shav_engine> e(new shav_engine);
    e->cfg = c;
    e->version = c.version ? c.version : "shield-av-1";
    e->audio_weights = sdup(c.audio_weights);
    e->video_weights = sdup(c.video_weights);
    e->cpu_isa_s = sdup(c.cpu_isa);
    std::string backend = c.video_backend && *c.video_backend ? c.video_backend : "auto";
    if (backend != "auto" && backend != "cuda" && backend != "cpu" && backend != "none")
        return (fail("video_backend must be auto|cuda|cpu|none"), nullptr);
    // ---- audio
    if (!e->audio_weights.empty()) {
        ThreadName tn("shav-apool");
        const char* mode = c.audio_mode && *c.audio_mode ? c.audio_mode : "exact";
        e->amodel = shield_model_load_mode(e->audio_weights.c_str(), c.audio_workers, mode);
        if (!e->amodel) return (fail(std::string("audio model: ") + shield_last_error()), nullptr);
        e->audio_mode_s = shield_model_mode(e->amodel);
        e->audio_fp = shield_model_fingerprint(e->amodel);
        e->audio_rev = shield_compute_revision();
        e->load_log += "audio " + e->audio_mode_s + " ok; ";
    }
    // ---- video
    if (!e->video_weights.empty() && backend != "none") {
        e->cascade = shv_cascade_load(nullptr);
        if (!e->cascade) return (fail(std::string("cascade: ") + shv_prep_last_error()), nullptr);
        if (backend == "auto" || backend == "cuda") {
            if (c.cuda_blocking_sync) cuda_blocking_sync(c.cuda_device, e->load_log);
            shv_cuda_options o;
            shv_cuda_default_options(&o);
            o.token_slots = c.max_sessions * NSLOT;
            o.max_delay_us = c.cuda_max_delay_us;
            o.max_frames = c.cuda_max_frames;
            o.max_windows = c.cuda_max_windows;
            std::vector<std::string> modes;
            if (c.video_mode && *c.video_mode) modes.push_back(c.video_mode);
            else modes = {"fp16", "fp32"};
            for (const auto& m : modes) {
                ThreadName tn("shav-gpu");
                e->cuda = shv_cuda_load_ex(e->video_weights.c_str(), m.c_str(), c.cuda_device, &o);
                if (e->cuda) {
                    e->load_log += "cuda " + m + " ok; ";
                    break;
                }
                e->load_log += "cuda " + m + " failed: " + shv_cuda_last_error() + "; ";
            }
            if (e->cuda) {
                e->backend = 1;
                e->video_mode_s = shv_cuda_mode(e->cuda);
                e->video_fp = shv_cuda_fingerprint(e->cuda);
                e->video_rev = shv_cuda_compute_revision();
                e->vthr = shv_cuda_threshold(e->cuda);
                e->device_bytes = shv_cuda_device_bytes(e->cuda);
            } else if (backend == "cuda") {
                return (fail("video cuda backend: " + e->load_log), nullptr);
            }
        }
        if (!e->backend) {
            const char* m = (backend == "cpu" && c.video_mode && *c.video_mode) ? c.video_mode : "exact";
            ThreadName tn("shav-cpupool");
            e->cpu = shv_cpu_load_ex(e->video_weights.c_str(), m, c.cpu_video_threads,
                                     e->cpu_isa_s.empty() ? nullptr : e->cpu_isa_s.c_str());
            if (!e->cpu) return (fail(e->load_log + "video cpu: " + shv_cpu_last_error()), nullptr);
            e->load_log += std::string("cpu ") + m + " ok; ";
            e->backend = 2;
            e->video_mode_s = shv_cpu_mode(e->cpu);
            e->video_fp = shv_cpu_fingerprint(e->cpu);
            e->video_rev = shv_cpu_revision();
            e->vthr = shv_cpu_threshold(e->cpu);
        }
        if (!std::isfinite(e->vthr)) return (fail("video threshold missing in <weights>.json"), nullptr);
        e->video_backend_s = e->backend == 1 ? "cuda" : "cpu";
    } else {
        e->video_backend_s = "none";
    }
    for (int b = c.max_sessions - 1; b >= 0; --b) e->free_blocks.push_back(b);
    // ---- threads
    for (int i = 0; i < c.audio_workers; ++i)
        e->adrv.emplace_back([p = e.get()] {
            pthread_setname_np(pthread_self(), "shav-adrv");
            p->audio_loop();
        });
    if (e->backend == 1)
        e->vdone = std::thread([p = e.get()] {
            pthread_setname_np(pthread_self(), "shav-vdone");
            p->video_done_loop();
        });
    if (e->backend == 2)
        for (int i = 0; i < c.cpu_video_workers; ++i)
            e->cw.emplace_back([p = e.get()] {
                pthread_setname_np(pthread_self(), "shav-vcpu");
                p->cpu_loop();
            });
    g_err.clear();
    return e.release();
}

void shav_engine_free(shav_engine* e) {
    if (!e) return;
    {
        std::lock_guard<std::mutex> a(e->aq_mu), b(e->vq_mu), c(e->cq_mu);
        e->stopping = true;
    }
    e->aq_cv.notify_all();
    e->vq_cv.notify_all();
    e->cq_cv.notify_all();
    for (auto& t : e->adrv) t.join();
    if (e->vdone.joinable()) e->vdone.join();
    for (auto& t : e->cw) t.join();
    if (e->cuda) shv_cuda_free(e->cuda);
    for (uint8_t* b : e->pool_all) {
        if (e->backend == 1) shv_cuda_host_free(b);
        else std::free(b);
    }
    if (e->cpu) shv_cpu_free(e->cpu);
    if (e->amodel) shield_model_free(e->amodel);
    if (e->cascade) shv_cascade_free(e->cascade);
    delete e;
}

int shav_engine_get_info(const shav_engine* e, shav_engine_info* i) {
    if (!e || !i) return fail("null argument");
    i->version = e->version.c_str();
    i->audio_mode = e->audio_mode_s.c_str();
    i->audio_fingerprint = e->audio_fp.c_str();
    i->audio_revision = e->audio_rev.c_str();
    i->video_backend = e->video_backend_s.c_str();
    i->video_mode = e->video_mode_s.c_str();
    i->video_fingerprint = e->video_fp.c_str();
    i->video_revision = e->video_rev.c_str();
    i->video_threshold = e->vthr;
    i->load_log = e->load_log.c_str();
    return 0;
}

int shav_engine_get_stats(const shav_engine* e_, shav_engine_stats* st) {
    if (!e_ || !st) return fail("null argument");
    shav_engine* e = const_cast<shav_engine*>(e_);
    std::memset(st, 0, sizeof *st);
    st->sessions = e->n_sessions.load();
    st->audio_jobs = e->n_audio.load();
    st->video_jobs = e->n_video.load();
    st->video_frames_encoded = e->n_enc.load();
    st->video_frames_reused = e->n_reuse.load();
    st->video_dropped = e->n_drop.load();
    {
        std::lock_guard<std::mutex> lk(e->aq_mu);
        st->audio_queue_depth = int64_t(e->aq.size());
    }
    {
        std::lock_guard<std::mutex> lk(e->cq_mu);
        st->video_queue_depth = int64_t(e->cq.size());
    }
    {
        std::lock_guard<std::mutex> lk(e->pool_mu);
        st->crop_buffers = e->pool_all.size();
        st->crop_buffer_bytes = e->pool_all.size() * 20 * FB;
    }
    st->video_device_bytes = e->device_bytes;
    return 0;
}

int shav_engine_video_predict(shav_engine* e, const uint8_t* clips, size_t b, float* logits) {
    if (!e || (b && (!clips || !logits))) return fail("null argument");
    if (e->backend == 1) return shv_cuda_predict(e->cuda, clips, b, logits) == 0 ? 0 : fail(shv_cuda_last_error());
    if (e->backend == 2) return shv_cpu_predict(e->cpu, clips, b, logits) == 0 ? 0 : fail(shv_cpu_last_error());
    return fail("engine has no video model");
}

shav_session* shav_session_create(shav_engine* e, const shav_session_config* cfg_in) {
    if (!e) return (fail("null engine"), nullptr);
    shav_session_config dc;
    shav_session_default_config(&dc);
    const shav_session_config& c = cfg_in ? *cfg_in : dc;
    if (!c.participant_id || !*c.participant_id) return (fail("participant_id required"), nullptr);
    if (!std::isfinite(c.audio_threshold) || c.audio_threshold < 0 || c.audio_threshold > 1)
        return (fail("audio_threshold must be in [0,1]"), nullptr);
    if (c.audio_alert_k < 1 || c.audio_alert_n < c.audio_alert_k || c.audio_alert_n > 100000 || c.video_alert_k < 1 ||
        c.video_alert_n < c.video_alert_k || c.video_alert_n > 100000)
        return (fail("alert k/n must satisfy 1 <= k <= n <= 100000"), nullptr);
    if (c.policy < 0 || c.policy > 3) return (fail("unknown policy"), nullptr);
    if (!(c.video_wait_s >= 0) || !std::isfinite(c.video_wait_s)) return (fail("video_wait_s must be >= 0"), nullptr);
    if (!std::isnan(c.audio_wait_s) && !(c.audio_wait_s >= 0))
        return (fail("audio_wait_s must be NaN (= video_wait_s), >= 0, or +inf (rule off)"), nullptr);
    if (c.video_store < 0 || c.video_store > 2) return (fail("video_store must be 0..2"), nullptr);
    if (c.video_max_pending < 0 || c.video_max_pending > 64) return (fail("video_max_pending must be 0..64"), nullptr);
    if (c.audio_lowpass != 0 && c.audio_lowpass != 1) return (fail("audio_lowpass must be 0 or 1"), nullptr);
    if (!c.audio_enabled && !c.video_enabled) return (fail("a session needs at least one modality"), nullptr);
    if (std::isfinite(c.video_start) && c.video_first_hop < 0) return (fail("video_first_hop must be >= 0"), nullptr);
    std::unique_ptr<shav_session> s(new shav_session);
    s->e = e;
    s->cfg = c;
    s->id = c.participant_id;
    s->cfg.participant_id = s->id.c_str();
    s->audio_on = c.audio_enabled != 0;
    s->video_on = c.video_enabled != 0 && e->backend != 0;
    s->audio_wait = std::isnan(c.audio_wait_s) ? c.video_wait_s : c.audio_wait_s;
    if (s->audio_on) {
        s->stream = shield_stream_create(e->amodel, s->id.c_str(), c.audio_threshold, e->version.c_str(), c.audio_alert_k,
                                         c.audio_alert_n);
        if (!s->stream) return (fail(std::string("audio stream: ") + shield_last_error()), nullptr);
        if (c.audio_lowpass && shield_stream_set_lowpass(s->stream, 1) != 0) {
            fail(std::string("audio low-pass: ") + shield_last_error());
            shield_stream_free(s->stream);
            return nullptr;
        }
        if (e->cfg.audio_cache_level >= 0 && shield_stream_set_cache_level(s->stream, e->cfg.audio_cache_level) != 0) {
            fail(std::string("audio cache level: ") + shield_last_error());
            shield_stream_free(s->stream);
            return nullptr;
        }
        s->hopbuf.resize(HOP);
        s->stream_bytes.store(shield_stream_bytes(s->stream));
    }
    if (s->video_on) {
        if (e->backend == 1) {
            std::lock_guard<std::mutex> lk(e->slot_mu);
            if (e->free_blocks.empty()) {
                if (s->stream) shield_stream_free(s->stream);
                return (fail("max_sessions reached (device token slots)"), nullptr);
            }
            s->slot_base = e->free_blocks.back() * NSLOT;
            e->free_blocks.pop_back();
        } else {
            s->host_tok = static_cast<float*>(std::aligned_alloc(64, NSLOT * TF * sizeof(float)));
            if (!s->host_tok) {
                if (s->stream) shield_stream_free(s->stream);
                return (fail("out of memory"), nullptr);
            }
        }
    }
    // one crop staging buffer per video session up front (a session holds one while its window is in flight);
    // further ones (pending windows) are allocated on demand
    if (s->video_on) e->reserve_buf();
    e->n_sessions.fetch_add(1);
    g_err.clear();
    return s.release();
}

static void drain_session(shav_session* s) {
    // drop queued (not submitted) jobs, then wait for the in-flight ones; their results are discarded
    {
        std::lock_guard<std::mutex> lk(s->mu);
        s->discard = true;
    }
    std::vector<AudioJob*> ad;
    {
        std::lock_guard<std::mutex> lk(s->amu);
        ad.assign(s->a_pending.begin(), s->a_pending.end());
        s->a_pending.clear();
    }
    for (AudioJob* j : ad) {
        delete j;
        s->job_done();
    }
    std::vector<VideoJob*> vd;
    {
        std::lock_guard<std::mutex> lk(s->jmu);
        vd.assign(s->v_pending.begin(), s->v_pending.end());
        s->v_pending.clear();
    }
    for (VideoJob* j : vd) {
        s->e->put_buf(j->crops);
        s->crops_held.fetch_sub(1);
        delete j;
        s->job_done();
    }
    std::unique_lock<std::mutex> lk(s->imu);
    s->icv.wait(lk, [&] { return s->inflight == 0; });
}

void shav_session_free(shav_session* s) {
    if (!s) return;
    drain_session(s);
    if (s->stream) shield_stream_free(s->stream);
    if (s->part) shv_participant_free(s->part);
    if (s->host_tok) std::free(s->host_tok);
    if (s->slot_base >= 0) {
        std::lock_guard<std::mutex> lk(s->e->slot_mu);
        s->e->free_blocks.push_back(s->slot_base / NSLOT);
    }
    s->e->n_sessions.fetch_sub(1);
    delete s;
}

int shav_session_reset(shav_session* s) {
    if (!s) return fail("null session");
    drain_session(s);
    std::lock_guard<std::mutex> l1(s->vmu), l2(s->amu), l3(s->mu);
    if (s->stream) shield_stream_reset(s->stream);
    s->fill = 0;
    s->audio_hops = 0;
    s->a_busy = false;
    if (s->part) shv_participant_free(s->part);
    s->part = nullptr;
    s->last_t = -std::numeric_limits<double>::infinity();
    s->next_window = 0;
    for (auto& c : s->cache) c = CacheEntry();
    s->stamp = 0;
    s->video_started.store(false);
    s->audio_started.store(false);
    s->first_hop.store(-1);
    s->cache_gen.store(0);
    s->poison_gen.store(-1);
    s->v_busy = false;
    s->hops.clear();
    s->out.clear();
    s->v_hist.clear();
    s->next_emit = s->v_settle_from = s->audio_clock = 0;
    s->finished = false;
    s->discard = false;
    s->emitted = 0;
    s->st_vres = s->st_vscored = s->st_pushed = s->st_sampled = s->st_enc = s->st_reuse = 0;
    return 0;
}

static int push_audio_impl(shav_session* s, const float* pcm, size_t n) {
    std::vector<AudioJob*> ready;
    int64_t last_hop = -1;
    if (n) s->audio_started.store(true);
    {
        std::lock_guard<std::mutex> lk(s->amu);
        size_t done = 0;
        while (done < n) {
            const size_t take = std::min(n - done, size_t(HOP) - s->fill);
            std::memcpy(s->hopbuf.data() + s->fill, pcm + done, take * sizeof(float));
            s->fill += take;
            done += take;
            if (s->fill == size_t(HOP)) {
                AudioJob* j = new AudioJob;
                j->s = s;
                j->hop = s->audio_hops++;
                j->pcm.swap(s->hopbuf);
                s->hopbuf.resize(HOP);
                j->t_input = mono();
                s->fill = 0;
                s->job_add();
                if (!s->a_busy) {
                    s->a_busy = true;
                    ready.push_back(j);
                } else {
                    s->a_pending.push_back(j);
                }
                last_hop = j->hop;
            }
        }
    }
    for (AudioJob* j : ready) enqueue_audio(s->e, j);
    if (last_hop >= 0) {
        std::lock_guard<std::mutex> lk(s->mu);
        for (int64_t h = std::max(s->audio_clock, s->next_emit); h <= last_hop; ++h) s->hop_state(h);
        s->audio_clock = last_hop + 1;
        s->settle_video(last_hop, false);
        s->try_emit();
    }
    return 0;
}

int shav_session_push_audio(shav_session* s, const float* pcm, size_t n) {
    if (!s) return fail("null session");
    if (!s->audio_on) return fail("audio disabled for this session");
    if (n && !pcm) return fail("null pcm");
    {
        std::lock_guard<std::mutex> lk(s->mu);
        if (s->finished) return fail("session finished; reset it first");
    }
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(pcm[i]) || std::fabs(pcm[i]) > 1.0f) return fail("audio must be finite float32 in [-1,1]");
    return push_audio_impl(s, pcm, n);
}

int shav_session_push_audio_s16(shav_session* s, const int16_t* pcm, size_t n) {
    if (!s) return fail("null session");
    if (!s->audio_on) return fail("audio disabled for this session");
    if (n && !pcm) return fail("null pcm");
    {
        std::lock_guard<std::mutex> lk(s->mu);
        if (s->finished) return fail("session finished; reset it first");
    }
    thread_local std::vector<float> buf;
    buf.resize(n);
    for (size_t i = 0; i < n; ++i) buf[i] = float(pcm[i]) / 32768.0f;
    return push_audio_impl(s, buf.data(), n);
}

namespace {

// token-cache lookup / allocation for one READY window (under vmu)
void cache_plan(shav_session* s, const int64_t* targets, const shv_rect* rects, VideoJob* j, bool use_cache) {
    if (s->poison_gen.load() >= s->cache_gen.load()) {
        for (auto& c : s->cache) c = CacheEntry();
        s->cache_gen.fetch_add(1);
    }
    j->gen = s->cache_gen.load();
    bool used[NSLOT] = {false};
    int n_new = 0, n_hit = 0;
    for (int i = 0; i < 20; ++i) {
        j->entry[i] = -1;
        if (!use_cache) continue;
        for (int c = 0; c < NSLOT; ++c) {
            const CacheEntry& ce = s->cache[c];
            if (ce.valid && ce.k == targets[i] && ce.r.x == rects[i].x && ce.r.y == rects[i].y && ce.r.w == rects[i].w &&
                ce.r.h == rects[i].h) {
                j->entry[i] = c;
                used[c] = true;
                break;
            }
        }
    }
    for (int i = 0; i < 20; ++i) {
        if (j->entry[i] >= 0) {
            ++n_hit;
            j->wframes[i] = -1;
            continue;
        }
        // free entry: invalid first, then frames no later window can use (k < targets[0]), then the oldest
        int best = -1;
        long long best_key = 0;
        for (int c = 0; c < NSLOT; ++c) {
            if (used[c]) continue;
            const CacheEntry& ce = s->cache[c];
            const long long key = !ce.valid ? -3 : (ce.k < targets[0] ? -2 : (long long)ce.stamp);
            if (best < 0 || key < best_key) best = c, best_key = key;
        }
        used[best] = true;
        j->entry[i] = best;
        CacheEntry& ce = s->cache[best];
        ce.valid = true;
        ce.k = targets[i];
        ce.r = rects[i];
        ce.stamp = ++s->stamp;
        j->wframes[i] = n_new;
        ++n_new;
    }
    j->n_new = n_new;
    j->n_hit = n_hit;
    j->has_hits = n_hit > 0;
    int f = 0;
    for (int i = 0; i < 20; ++i) {
        const int32_t slot = s->slot_base >= 0 ? int32_t(s->slot_base + j->entry[i]) : int32_t(j->entry[i]);
        j->wslots[i] = slot;
        if (j->wframes[i] >= 0) j->store[f++] = slot;
    }
}

void invalidate_entries(shav_session* s, const VideoJob* j) {
    for (int i = 0; i < 20; ++i)
        if (j->wframes[i] >= 0 && j->entry[i] >= 0) s->cache[j->entry[i]] = CacheEntry();
}

// one resolved participant window (under vmu)
void process_window(shav_session* s, int64_t w, int status, double t_in) {
    shav_engine* e = s->e;
    const int64_t h = s->first_hop.load() + w;
    s->st_vres.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(s->mu);
        if (s->discard || h < s->next_emit) return;
        HopState& hs = s->hop_state(h);
        if (hs.v_final) return;   // forced earlier (lag / ended)
        if (status != SHV_WIN_READY) {
            hs.v.t_input = t_in;
            s->video_final(hs, SHAV_UNAVAILABLE, SHAV_R_GAP);
            s->try_emit();
            return;
        }
        hs.v_resolving = true;   // from here on the audio clock must not force this hop (lag / ended)
        if (std::isfinite(s->cfg.video_start)) {
            hs.v.window_start = s->vstart + 2.0 * double(w);
            hs.v.window_end = hs.v.window_start + 10.0;
        }
    }
    // backpressure: at most video_max_pending windows behind the in-flight one
    {
        std::lock_guard<std::mutex> lk(s->jmu);
        if (s->v_busy && int(s->v_pending.size()) >= s->cfg.video_max_pending) {
            e->n_drop.fetch_add(1);
            std::lock_guard<std::mutex> l2(s->mu);
            HopState& hs = s->hop_state(h);
            hs.v.t_input = t_in;
            s->video_final(hs, SHAV_UNAVAILABLE, SHAV_R_OVERLOAD);
            s->try_emit();
            return;
        }
    }
    shv_window_info info;
    int64_t targets[20];
    double times[20];
    VideoJob* j = new VideoJob;
    j->s = s;
    j->hop = h;
    j->window = w;
    j->t_input = t_in;
    bool ok = shv_participant_window_info(s->part, w, &info, targets, times) == 0;
    if (ok) {
        j->anchors = info.anchors;
        cache_plan(s, targets, info.rects, j, s->cfg.video_cache != 0);
        j->crops = e->get_buf();
        ok = j->crops != nullptr;
        if (ok) s->crops_held.fetch_add(1);
        for (int i = 0; ok && i < 20; ++i)
            if (j->wframes[i] >= 0)
                ok = shv_participant_crop(s->part, targets[i], info.rects[i], j->crops + size_t(j->wframes[i]) * FB) == 0;
        if (ok && s->cfg.crop_tap) {
            s->tapbuf.resize(20 * FB);
            for (int i = 0; ok && i < 20; ++i) {
                if (j->wframes[i] >= 0)
                    std::memcpy(s->tapbuf.data() + size_t(i) * FB, j->crops + size_t(j->wframes[i]) * FB, FB);
                else
                    ok = shv_participant_crop(s->part, targets[i], info.rects[i], s->tapbuf.data() + size_t(i) * FB) == 0;
            }
            if (ok) {
                shav_window_view v;
                v.hop = h;
                v.window = w;
                v.anchors = info.anchors;
                for (int i = 0; i < 20; ++i) {
                    v.targets[i] = targets[i];
                    v.times[i] = times[i];
                    v.rects[i][0] = info.rects[i].x;
                    v.rects[i][1] = info.rects[i].y;
                    v.rects[i][2] = info.rects[i].w;
                    v.rects[i][3] = info.rects[i].h;
                }
                v.crops = s->tapbuf.data();
                s->cfg.crop_tap(&v, s->cfg.crop_tap_user);
            }
        }
        if (!ok) invalidate_entries(s, j);
    }
    if (!ok) {
        e->put_buf(j->crops);
        if (j->crops) s->crops_held.fetch_sub(1);
        std::lock_guard<std::mutex> lk(s->mu);
        HopState& hs = s->hop_state(h);
        hs.v.t_input = t_in;
        s->video_final(hs, SHAV_UNAVAILABLE, SHAV_R_FAILED);
        s->try_emit();
        delete j;
        return;
    }
    s->st_enc.fetch_add(j->n_new);
    s->st_reuse.fetch_add(j->n_hit);
    e->n_enc.fetch_add(j->n_new);
    e->n_reuse.fetch_add(j->n_hit);
    s->job_add();
    bool now = false;
    {
        std::lock_guard<std::mutex> lk(s->jmu);
        if (!s->v_busy) {
            s->v_busy = true;
            now = true;
        } else {
            s->v_pending.push_back(j);
        }
    }
    if (now) e->submit_video(j);
}

int push_video_impl(shav_session* s, double t, const shv_rgb_image* rgb, const shv_i420_image* yuv) {
    if (!s) return fail("null session");
    if (!s->video_on) return fail("video disabled for this session (or the engine has no video model)");
    if (!std::isfinite(t)) return fail("video timestamp must be finite");
    const bool ref_mode = std::isfinite(s->cfg.video_start);
    if (!ref_mode && t < 0) return fail("video timestamp must be >= 0 (session seconds)");
    const double t_in = mono();
    std::lock_guard<std::mutex> lk(s->vmu);
    {
        std::lock_guard<std::mutex> l2(s->mu);
        if (s->finished) return fail("session finished; reset it first");
    }
    if (!(t > s->last_t)) return fail("video timestamps must be strictly increasing");
    if (!s->part) {
        int64_t fh;
        double start;
        if (ref_mode) {
            start = s->cfg.video_start;
            fh = s->cfg.video_first_hop;
        } else {
            // first window start: the smallest multiple of 2 s (session clock) that the first frame can still serve
            const int64_t k0 = std::max<int64_t>(0, int64_t(4 * std::ceil((t - 0.25) / 2.0)));
            start = double(k0) / 2.0 - t;
            fh = k0 / 4 + 4;
        }
        s->part = shv_participant_create(s->e->cascade, start, 4);
        if (!s->part) return fail(std::string("video participant: ") + shv_prep_last_error());
        shv_participant_set_store(s->part, s->cfg.video_store);
        s->t0 = t;
        s->vstart = start;
        s->first_hop.store(fh);
        s->video_started.store(true);
        std::lock_guard<std::mutex> l2(s->mu);
        for (int64_t h = std::max(s->v_settle_from, s->next_emit); h < fh; ++h) {
            HopState& hs = s->hop_state(h);
            if (!hs.v_final) s->video_final(hs, SHAV_COLLECTING, SHAV_R_WARMUP);
        }
        s->try_emit();
    }
    shv_push_result res;
    const int rc = rgb ? shv_participant_push_rgb(s->part, t, rgb, &res) : shv_participant_push_i420(s->part, t, yuv, &res);
    if (rc != 0) return fail(std::string("video push: ") + shv_prep_last_error());
    s->last_t = t;
    s->st_pushed.fetch_add(1);
    if (res.selected_target >= 0) s->st_sampled.fetch_add(1);
    if (res.ready_window >= 0) {
        for (int64_t w = s->next_window; w <= res.ready_window; ++w) {
            // a window resolved together with a later one contains a missing target (its frames may be released)
            const int st = w == res.ready_window ? res.ready_status : SHV_WIN_UNAVAILABLE;
            process_window(s, w, st, t_in);
        }
        s->next_window = res.ready_window + 1;
        // every crop of the resolved windows was extracted above: keep only the frames later windows need
        // (peak 20 instead of 23 cached frames per participant)
        shv_participant_release_before(s->part, res.ready_window + 1);
    }
    // audio lag rule (video clock).  Reference mode maps the frame time onto the session hop timeline: participant
    // window w = hop first_hop + w ends at (t - t0) = vstart + 2w + 10.
    if (s->audio_on && std::isfinite(s->audio_wait)) {
        const int64_t fh = s->first_hop.load();
        const double tv = ref_mode ? (t - s->t0) - s->vstart + 2.0 * double(fh - 4) : t;
        std::lock_guard<std::mutex> l2(s->mu);
        s->video_clock(tv, fh + s->next_window - 1);
        s->try_emit();
    }
    return 0;
}

}  // namespace

int shav_session_push_video_rgb(shav_session* s, double t, const uint8_t* rgb, int w, int h, ptrdiff_t stride) {
    if (!rgb || w <= 0 || h <= 0 || stride < ptrdiff_t(3) * w) return fail("bad RGB frame");
    shv_rgb_image im{rgb, w, h, stride};
    return push_video_impl(s, t, &im, nullptr);
}

int shav_session_push_video_i420(shav_session* s, double t, const uint8_t* y, ptrdiff_t ys, const uint8_t* u, ptrdiff_t us,
                                 const uint8_t* v, ptrdiff_t vs, int w, int h) {
    if (!y || !u || !v || w <= 0 || h <= 0 || ys < w || us < (w + 1) / 2 || vs < (w + 1) / 2) return fail("bad I420 frame");
    shv_i420_image im{y, u, v, ys, us, vs, w, h};
    return push_video_impl(s, t, nullptr, &im);
}

int shav_session_wait_idle(shav_session* s, double timeout_s) {
    if (!s) return fail("null session");
    std::unique_lock<std::mutex> lk(s->imu);
    if (timeout_s < 0) {
        s->icv.wait(lk, [&] { return s->inflight == 0; });
        return 0;
    }
    const bool ok = s->icv.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] { return s->inflight == 0; });
    return ok ? 0 : fail("timeout");
}

int shav_session_finish(shav_session* s, double timeout_s) {
    if (!s) return fail("null session");
    {
        std::lock_guard<std::mutex> l1(s->vmu), l2(s->mu);
        s->finished = true;
        s->settle_end();
        s->try_emit();
    }
    if (shav_session_wait_idle(s, timeout_s) != 0) return -1;
    std::lock_guard<std::mutex> l2(s->mu);
    s->settle_end();
    s->try_emit();
    return 0;
}

size_t shav_session_poll(shav_session* s, shav_hop_result* out, size_t cap) {
    if (!s || !out) return 0;
    std::lock_guard<std::mutex> lk(s->mu);
    size_t n = 0;
    while (n < cap && !s->out.empty()) {
        out[n++] = s->out.front();
        s->out.pop_front();
    }
    return n;
}

const char* shav_session_id(const shav_session* s) { return s ? s->id.c_str() : nullptr; }

int64_t shav_session_video_first_hop(const shav_session* s) { return s && s->video_started.load() ? s->first_hop.load() : -1; }

int shav_session_get_stats(const shav_session* s_, shav_session_stats* st) {
    if (!s_ || !st) return fail("null argument");
    shav_session* s = const_cast<shav_session*>(s_);
    std::memset(st, 0, sizeof *st);
    {
        std::lock_guard<std::mutex> lk(s->mu);
        st->hops_emitted = s->emitted;
        st->bytes_other = sizeof(shav_session) + s->out.size() * sizeof(shav_hop_result) + s->hops.size() * (sizeof(HopState) + 48);
    }
    {
        std::lock_guard<std::mutex> lk(s->amu);
        st->audio_hops = s->audio_hops;
        st->bytes_audio = s->stream ? s->stream_bytes.load() + s->hopbuf.capacity() * sizeof(float) : 0;
    }
    {
        std::lock_guard<std::mutex> lk(s->vmu);
        st->bytes_video_frames = s->part ? shv_participant_bytes(s->part) : 0;
        st->bytes_video_frames += s->tapbuf.capacity();
    }
    st->video_windows_resolved = s->st_vres.load();
    st->video_windows_scored = s->st_vscored.load();
    st->video_frames_pushed = s->st_pushed.load();
    st->video_frames_sampled = s->st_sampled.load();
    st->video_frames_encoded = s->st_enc.load();
    st->video_frames_reused = s->st_reuse.load();
    st->bytes_crops = size_t(std::max(0, s->crops_held.load())) * 20 * FB;
    st->bytes_tokens_host = s->host_tok ? NSLOT * TF * sizeof(float) : 0;
    st->bytes_tokens_device = s->slot_base >= 0 ? NSLOT * TF * sizeof(float) : 0;
    st->bytes_total = st->bytes_audio + st->bytes_video_frames + st->bytes_crops + st->bytes_tokens_host +
                      st->bytes_tokens_device + st->bytes_other;
    return 0;
}

size_t shav_session_bytes(const shav_session* s) {
    shav_session_stats st;
    return shav_session_get_stats(s, &st) == 0 ? st.bytes_total : 0;
}

static void jnum(std::string& o, double v, const char* fmt = "%.17g") {
    if (!std::isfinite(v)) {
        o += "null";
        return;
    }
    char b[64];
    std::snprintf(b, sizeof b, fmt, v);
    o += b;
}
static void jpart(std::string& o, const shav_modality_result& m, bool audio) {
    o += "{\"status\":\"";
    o += shav_status_name(m.status);
    o += "\",\"reason\":\"";
    o += shav_reason_name(m.reason);
    o += "\",\"score\":";
    jnum(o, m.score);
    o += ",\"logit\":";
    jnum(o, double(m.logit), "%.9g");
    o += ",\"decision\":";
    o += m.decision < 0 ? "null" : std::to_string(m.decision);
    o += ",\"alert\":";
    o += m.alert < 0 ? "null" : std::to_string(m.alert);
    o += ",\"window\":[";
    jnum(o, m.window_start, "%.6f");
    o += ",";
    jnum(o, m.window_end, "%.6f");
    o += "]";
    if (audio) {
        o += ",\"valid_speech_sec\":";
        jnum(o, m.valid_speech_sec, "%.2f");
    } else {
        o += ",\"face_detection_anchors\":";
        o += m.face_detection_anchors < 0 ? "null" : std::to_string(m.face_detection_anchors);
        o += ",\"frames_encoded\":";
        o += m.frames_encoded < 0 ? "null" : std::to_string(m.frames_encoded);
        o += ",\"frames_reused\":";
        o += m.frames_reused < 0 ? "null" : std::to_string(m.frames_reused);
    }
    o += ",\"latency_ms\":";
    jnum(o, (m.t_done - m.t_input) * 1e3, "%.3f");
    o += "}";
}

int shav_result_json(const shav_hop_result* r, const char* pid, char* buf, size_t cap) {
    if (!r) return -1;
    std::string o = "{";
    if (pid) {
        o += "\"participant\":\"";
        // RFC 8259 string escaping: '"' and '\\' backslash-escaped, control characters U+0000..U+001F as \b \f \n \r \t
        // or \u00XX (never dropped); all other bytes (UTF-8 included) are copied as they are.
        for (const char* p = pid; *p; ++p) {
            const unsigned char ch = static_cast<unsigned char>(*p);
            switch (ch) {
                case '"': o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\b': o += "\\b"; break;
                case '\f': o += "\\f"; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                default:
                    if (ch < 0x20) {
                        char u[8];
                        std::snprintf(u, sizeof u, "\\u%04x", unsigned(ch));
                        o += u;
                    } else {
                        o += char(ch);
                    }
            }
        }
        o += "\",";
    }
    o += "\"hop\":" + std::to_string(r->hop) + ",\"hop_end\":";
    jnum(o, r->hop_end, "%.3f");
    static const char* pol[] = {"either", "both", "audio", "video"};
    o += ",\"policy\":\"";
    o += (r->policy >= 0 && r->policy <= 3) ? pol[r->policy] : "?";
    o += "\",\"alert\":";
    o += r->alert < 0 ? "null" : std::to_string(r->alert);
    o += ",\"audio\":";
    jpart(o, r->audio, true);
    o += ",\"video\":";
    jpart(o, r->video, false);
    o += "}";
    if (buf && cap) {
        const size_t n = std::min(cap - 1, o.size());
        std::memcpy(buf, o.data(), n);
        buf[n] = 0;
    }
    return int(o.size());
}

}  // extern "C"
