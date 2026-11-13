"""ctypes wrapper of build/cpu/libshield_v_cpu.so (verification / benchmarking only; the engine itself has no Python).

    from bench.cpu_engine import CpuModel
    m = CpuModel(threads=8, isa=None, mode="exact", lib=..., weights=...)
    tok = m.encode(frames_u8)        # [n,180,320,3] uint8 -> [n,16,384] float32
    lg = m.classify(tokens)          # [b,20,16,384] float32 -> [b] float32
    lg = m.predict(clips_u8)         # [b,20,180,320,3] uint8 -> [b] float32
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build/cpu/libshield_v_cpu.so"
WEIGHTS = ROOT / "build/shield_v.weights"


class CpuModel:
    def __init__(self, threads=1, isa=None, mode="exact", lib=LIB, weights=WEIGHTS, experimental=False):
        self.lib = C.CDLL(str(lib))
        L = self.lib
        for name in ("shv_cpu_load_ex", "shv_cpu_load_experimental"):
            getattr(L, name).restype = C.c_void_p
            getattr(L, name).argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_char_p]
        L.shv_cpu_free.argtypes = [C.c_void_p]
        for name in ("shv_cpu_encode_frames", "shv_cpu_predict"):
            getattr(L, name).argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p]
            getattr(L, name).restype = C.c_int
        L.shv_cpu_classify.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p]
        L.shv_cpu_classify.restype = C.c_int
        L.shv_cpu_classify_ptrs.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.c_size_t, C.c_void_p]
        L.shv_cpu_classify_ptrs.restype = C.c_int
        L.shv_cpu_encode_frame_ptrs.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(C.c_void_p)]
        L.shv_cpu_encode_frame_ptrs.restype = C.c_int
        for name in ("shv_cpu_last_error", "shv_cpu_revision", "shv_cpu_version"):
            getattr(L, name).restype = C.c_char_p
        for name in ("shv_cpu_mode", "shv_cpu_fingerprint", "shv_cpu_isa"):
            getattr(L, name).restype = C.c_char_p
            getattr(L, name).argtypes = [C.c_void_p]
        L.shv_cpu_threshold.restype = C.c_double
        L.shv_cpu_threshold.argtypes = [C.c_void_p]
        L.shv_cpu_memory.argtypes = [C.c_void_p] + [C.POINTER(C.c_size_t)] * 3
        load = L.shv_cpu_load_experimental if experimental else L.shv_cpu_load_ex
        self.m = load(str(weights).encode(), (mode or "").encode(), int(threads), (isa or "").encode())
        if not self.m:
            raise RuntimeError(L.shv_cpu_last_error().decode())
        self.threads = threads

    def _check(self, rc):
        if rc:
            raise RuntimeError(self.lib.shv_cpu_last_error().decode())

    @property
    def mode(self):
        return self.lib.shv_cpu_mode(self.m).decode()

    @property
    def isa(self):
        return self.lib.shv_cpu_isa(self.m).decode()

    @property
    def fingerprint(self):
        return self.lib.shv_cpu_fingerprint(self.m).decode()

    @property
    def threshold(self):
        return self.lib.shv_cpu_threshold(self.m)

    def memory(self):
        a, b, c = C.c_size_t(), C.c_size_t(), C.c_size_t()
        self.lib.shv_cpu_memory(self.m, C.byref(a), C.byref(b), C.byref(c))
        return {"weight_bytes": a.value, "workspace_bytes": b.value, "slot_bytes": c.value}

    def encode(self, frames):
        f = np.ascontiguousarray(frames, dtype=np.uint8)
        assert f.ndim == 4 and f.shape[1:] == (180, 320, 3)
        out = np.empty((f.shape[0], 16, 384), np.float32)
        self._check(self.lib.shv_cpu_encode_frames(self.m, f.ctypes.data, f.shape[0], out.ctypes.data))
        return out

    def classify(self, tokens):
        t = np.ascontiguousarray(tokens, dtype=np.float32)
        assert t.ndim == 4 and t.shape[1:] == (20, 16, 384)
        out = np.empty(t.shape[0], np.float32)
        self._check(self.lib.shv_cpu_classify(self.m, t.ctypes.data, t.shape[0], out.ctypes.data))
        return out

    def classify_ptrs(self, frame_tokens):
        """frame_tokens: list of b*20 contiguous float32 [16,384] arrays."""
        arrs = [np.ascontiguousarray(a, dtype=np.float32) for a in frame_tokens]
        assert len(arrs) % 20 == 0
        ptrs = (C.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
        out = np.empty(len(arrs) // 20, np.float32)
        self._check(self.lib.shv_cpu_classify_ptrs(self.m, ptrs, len(arrs) // 20, out.ctypes.data))
        return out

    def predict(self, clips):
        c = np.ascontiguousarray(clips, dtype=np.uint8)
        assert c.ndim == 5 and c.shape[1:] == (20, 180, 320, 3)
        out = np.empty(c.shape[0], np.float32)
        self._check(self.lib.shv_cpu_predict(self.m, c.ctypes.data, c.shape[0], out.ctypes.data))
        return out

    def close(self):
        if self.m:
            self.lib.shv_cpu_free(self.m)
            self.m = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
