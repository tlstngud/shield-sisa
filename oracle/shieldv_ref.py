"""SHIELD-V oracle: the ShieldV function on CPU in pure PyTorch (float32 or float64).

The teammate's model (reference/hf_model/shield_v/model.py) needs fla (Triton, GPU only) for GatedDeltaNet and the
SISA Triton kernels on CUDA. This module re-implements exactly the same function with plain torch ops so that it runs
on CPU, in float32 (the native engine's "exact" tier) or float64 (to bound the oracle's own rounding error):

  frames [N,3,180,320] (RGB/255, ImageNet mean/std) -> torchvision MobileNetV3-Large .features -> [N,960,6,10]
  -> adaptive_avg_pool2d (4,4) (overlapping bins: rows [0,2) [1,3) [3,5) [4,6), cols [0,3) [2,5) [5,8) [7,10))
  -> flatten(2).transpose(1,2) (token = row*4+col) -> projection Linear(960,384,no bias) -> tokens [N,16,384]
  -> + (time[t] + row[r]) + col[c] -> [B,320,384] ++ readout -> [B,321,384]
  -> GDN-A (repeats 1) -> GDN-B (same mixer + norm twice, delta / sqrt(2), FFN once) -> SISA block (attn_impl torch)
  -> final RMSNorm(1e-6) on token 320 -> Linear(384,128) -> SiLU -> Linear(128,1) -> logit.

fla GatedDeltaNet semantics reproduced (fla 0.5.2, read from its source):
  q,k,v = SiLU(causal depthwise conv4 with bias (q_proj x)) ...   (ShortConvolution, activation 'silu')
  q,k  <- x / sqrt(sum(x^2) + 1e-6)     per head (fla l2norm_fwd default eps 1e-6, use_qk_l2norm_in_kernel=True)
  g     = -exp(A_log) * softplus(a_proj x + dt_bias)  (softplus threshold 20, as torch)     (use_gate_in_kernel)
  beta  = sigmoid(b_proj x)                                         (use_beta_sigmoid_in_kernel, allow_neg_eigval off)
  S_t = exp(g_t) S_{t-1} + k_t beta_t (v_t - (exp(g_t) S_{t-1})^T k_t)^T ;  o_t = S_t^T (q_t * K^-0.5)  (zero state)
  o    <- o * rsqrt(mean_V(o^2) + 1e-6) * o_norm.weight * SiLU(g_proj x)   (FusedRMSNormGated, eps = norm_eps 1e-6)
  out   = o_proj(o).
The output_gate_activation="silu" kwarg of the teammate's constructor is swallowed by **kwargs in fla; the gated norm
uses its default activation 'swish' (= SiLU), so the function is the same.

SISA (sisa @7903517, attn_impl "torch") reproduced op for op, but dtype-generic: the package casts g, Phi, the forget
logits, the rotated c/b and its RMSNorm to float32 even for float64 inputs, so its own module cannot give a float64
reference. In float32 this implementation performs the same operations as the package (checked by
oracle/validate_cpu.py).

API
  load_ref(checkpoint=DEFAULT_CKPT, dtype=torch.float32) -> ShieldVRef (eval, CPU, strict weights_only load)
  load_ref_from_weights(path="build/shield_v.weights", dtype) -> the same model from the exported SHLDV001 file
  read_shldv(path) -> {name: np.float32 array}
  normalize(raw_uint8, dtype)   [..., 180, 320, 3] uint8 -> [..., 3, 180, 320] (same ops as fast_data.prepare_clips)
  load_crop(sample_id) -> uint8 [20,180,320,3] from the reference crops
  model(clips [B,20,3,180,320]) -> logits [B];  model.encode_frames, model.cnn_map, model.classify_features
  model.classify_features(tokens, trace=dict) records every intermediate (see Trace keys in the code)
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SHIELD_DATA_DIR", "data")) / "shieldv_codec"
DEFAULT_CKPT = DATA / "hf_model/best.pt"
CROPS = DATA / "crops/orig_bt601"
MAP = DATA / "maps/orig.jsonl"
PUBLISHED = DATA / "results/orig_bt601.jsonl"
WEIGHTS = ROOT / "build/shield_v.weights"

NORM_EPS = 1e-6        # nn.RMSNorm(384, eps=1e-6) in GDNStage / final_norm; SISAConfig.norm_eps; GDN norm_eps
L2_EPS = 1e-6          # fla.modules.l2norm.l2norm_fwd default, used by use_qk_l2norm_in_kernel
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
D, H, DK, DV, CONV, DFF, DSTATE, NTOK = 384, 6, 64, 128, 4, 960, 32, 321
PARAMETERS = 9884507


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def crop_key(sample_id: str) -> str:
    """File stem of a reference crop (reference/codec_exp/score.py: sha256(sample_id)[:24])."""
    return hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def load_crop(sample_id: str, crops_dir=CROPS) -> np.ndarray:
    with np.load(Path(crops_dir) / (crop_key(sample_id) + ".npz")) as z:
        frames = z["frames"]
    if frames.shape != (20, 180, 320, 3) or frames.dtype != np.uint8:
        raise ValueError(f"bad crop {sample_id}: {frames.shape} {frames.dtype}")
    return frames


def read_map(path=MAP) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalize(raw, dtype=torch.float32) -> torch.Tensor:
    """uint8 [..., 180, 320, 3] (numpy or tensor) -> [..., 3, 180, 320] in `dtype`.

    float32: exactly fast_data.prepare_clips (x.float().div_(255); (x - mean) / std with float32 mean/std).
    float64: the same expression evaluated in float64 (mean/std are the float64 values of the decimal literals).
    """
    x = torch.as_tensor(np.ascontiguousarray(raw)) if isinstance(raw, np.ndarray) else raw
    if x.dtype != torch.uint8 or x.shape[-1] != 3:
        raise ValueError("expected uint8 [..., H, W, 3]")
    x = x.movedim(-1, -3).contiguous().to(dtype).div_(255)
    shape = [1] * (x.ndim - 3) + [3, 1, 1]
    mean = x.new_tensor(MEAN).view(shape)
    std = x.new_tensor(STD).view(shape)
    return (x - mean) / std


# ------------------------------------------------------------------------------------------------ building blocks
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """sisa.modeling.RMSNorm without the float32 cast (identical ops in float32)."""
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight


def l2norm_fla(x: torch.Tensor, eps: float = L2_EPS) -> torch.Tensor:
    """fla l2norm_fwd: y = x * (1 / sqrt(sum(x^2) + eps)) over the last dim."""
    return x * (1.0 / torch.sqrt((x * x).sum(-1, keepdim=True) + eps))


def gated_delta_rule_recurrent(q, k, v, g, beta, scale):
    """Exact recurrent gated delta rule (fla naive_recurrent_gated_delta_rule), in the inputs' dtype.

    q, k [B,T,H,K] (already L2-normalised), v [B,T,H,V], g (log decay) and beta [B,T,H]; zero initial state.
    """
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    q = q * scale
    decay = g.exp()
    state = q.new_zeros(b, h, dk, dv)
    out = q.new_empty(b, t, h, dv)
    for i in range(t):
        state = state * decay[:, i, :, None, None]
        kt = k[:, i]                                                       # [B,H,K]
        pred = (kt.unsqueeze(-2) @ state).squeeze(-2)                      # S^T k  [B,H,V]
        u = (v[:, i] - pred) * beta[:, i, :, None]
        state = state + kt.unsqueeze(-1) * u.unsqueeze(-2)
        out[:, i] = (q[:, i].unsqueeze(-2) @ state).squeeze(-2)
    return out


class ShortConv(nn.Conv1d):
    """fla ShortConvolution (depthwise causal conv, kernel 4, bias, SiLU); state-dict layout weight [C,1,4], bias [C]."""

    def __init__(self, channels: int, kernel_size: int = CONV):
        super().__init__(channels, channels, kernel_size, groups=channels, padding=kernel_size - 1, bias=True)

    def forward(self, x):                                                  # [B,T,C]
        t = x.shape[1]
        return F.silu(super().forward(x.transpose(1, 2))[..., :t].transpose(1, 2))


class GatedRMSNorm(nn.Module):
    """fla FusedRMSNormGated(head_v_dim, eps), activation swish: x * rstd * w * g * sigmoid(g)."""

    def __init__(self, dim: int = DV, eps: float = NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, o, gate):
        rstd = 1.0 / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + self.eps)
        return (o * rstd) * self.weight * F.silu(gate)


class GatedDeltaNetRef(nn.Module):
    """Parameter layout of fla.layers.GatedDeltaNet(hidden 384, 6 heads, head_dim 64, expand_v 2, conv 4 + bias)."""

    def __init__(self):
        super().__init__()
        kd, vd = H * DK, H * DV
        self.q_proj = nn.Linear(D, kd, bias=False)
        self.k_proj = nn.Linear(D, kd, bias=False)
        self.v_proj = nn.Linear(D, vd, bias=False)
        self.a_proj = nn.Linear(D, H, bias=False)
        self.b_proj = nn.Linear(D, H, bias=False)
        self.A_log = nn.Parameter(torch.zeros(H))
        self.dt_bias = nn.Parameter(torch.zeros(H))
        self.q_conv1d = ShortConv(kd)
        self.k_conv1d = ShortConv(kd)
        self.v_conv1d = ShortConv(vd)
        self.g_proj = nn.Linear(D, vd, bias=False)
        self.o_norm = GatedRMSNorm(DV, NORM_EPS)
        self.o_proj = nn.Linear(vd, D, bias=False)

    def forward(self, x, trace=None, tag=""):
        b, t, _ = x.shape
        q = self.q_conv1d(self.q_proj(x)).view(b, t, H, DK)
        k = self.k_conv1d(self.k_proj(x)).view(b, t, H, DK)
        v = self.v_conv1d(self.v_proj(x)).view(b, t, H, DV)
        qn, kn = l2norm_fla(q), l2norm_fla(k)
        g = -self.A_log.exp() * F.softplus(self.a_proj(x) + self.dt_bias)  # [B,T,H] log decay
        beta = torch.sigmoid(self.b_proj(x))
        o = gated_delta_rule_recurrent(qn, kn, v, g, beta, DK ** -0.5)
        gate = self.g_proj(x).view(b, t, H, DV)
        on = self.o_norm(o, gate)
        out = self.o_proj(on.reshape(b, t, H * DV))
        if trace is not None:
            for name, val in (("q_conv", q), ("k_conv", k), ("v_conv", v), ("q_l2", qn), ("k_l2", kn), ("g", g),
                              ("beta", beta), ("o_core", o), ("gate", gate), ("o_gated", on), ("out", out)):
                trace[f"{tag}.{name}"] = val
        return out


class SwiGLUV(nn.Module):
    """shield_v.model.SwiGLU: down(silu(gate x) * up x)."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D, DFF, bias=False)
        self.up = nn.Linear(D, DFF, bias=False)
        self.down = nn.Linear(DFF, D, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GDNStageRef(nn.Module):
    """shield_v.model.GDNStage: for r in repeats: h = h + mixer(norm(h)) / sqrt(repeats); h + ffn(ffn_norm(h))."""

    def __init__(self, repeats: int):
        super().__init__()
        self.repeats = repeats
        self.norm = nn.RMSNorm(D, eps=NORM_EPS)
        self.ffn_norm = nn.RMSNorm(D, eps=NORM_EPS)
        self.mixer = GatedDeltaNetRef()
        self.ffn = SwiGLUV()

    def forward(self, h, trace=None, tag=""):
        for r in range(self.repeats):
            hn = self.norm(h)
            delta = self.mixer(hn, trace, f"{tag}.rep{r}")
            h = h + delta / math.sqrt(self.repeats)
            if trace is not None:
                trace[f"{tag}.rep{r}.norm_in"] = hn
                trace[f"{tag}.rep{r}.h"] = h
        hn = self.ffn_norm(h)
        f = self.ffn(hn)
        if trace is not None:
            trace[f"{tag}.ffn_norm"] = hn
            trace[f"{tag}.ffn_out"] = f
        return h + f


def rotate_pairs(z, phi):
    """sisa.functional.rotate_pairs: rotate interleaved pairs (0,1),(2,3),... of z by phi."""
    z_even, z_odd = z[..., 0::2], z[..., 1::2]
    cos, sin = torch.cos(phi), torch.sin(phi)
    return torch.stack([z_even * cos - z_odd * sin, z_even * sin + z_odd * cos], dim=-1).flatten(-2)


def sisa_attention_ref(q, k, v, c, b, g, lam, fgate, block_q: int = 256):
    """sisa.functional.sisa_attention_torch / _attend_rows (blocked over queries like the package), any dtype."""
    seq_len = q.shape[2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    outs = []
    for start in range(0, seq_len, block_q):
        end = min(start + block_q, seq_len)
        qq, cc, gq, fq = q[:, :, start:end], c[:, :, start:end], g[:, :, start:end], fgate[:, :, start:end]
        kk, vv, bb, gk, fk = k[:, :, :end], v[:, :, :end], b[:, :, :end], g[:, :, :end], fgate[:, :, :end]
        pos_q = torch.arange(start, end, device=q.device)
        pos_k = torch.arange(end, device=q.device)
        causal = pos_k.unsqueeze(0) <= pos_q.unsqueeze(1)
        scores = torch.matmul(qq, kk.transpose(-1, -2)) * scale
        delta = (gq.unsqueeze(-1) - gk.unsqueeze(-2)).masked_fill(~causal, 0.0)
        ssm = torch.matmul(cc, bb.transpose(-1, -2))
        scores = scores + lam.view(1, -1, 1, 1) * torch.exp(delta) * ssm
        scores = scores + (fq.unsqueeze(-1) - fk.unsqueeze(-2))
        scores = scores.masked_fill(~causal, float("-inf"))
        outs.append(torch.matmul(torch.softmax(scores, dim=-1), vv))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)


class SISAAttentionRef(nn.Module):
    """Parameter layout of sisa.SISAAttention (d 384, 6 heads, d_head 64, d_state 32)."""

    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.W_O = nn.Linear(D, D, bias=False)
        self.W_B = nn.Linear(D, H * DSTATE, bias=False)
        self.W_C = nn.Linear(D, H * DSTATE, bias=False)
        self.alpha_proj = nn.Linear(D, H, bias=True)
        self.theta_proj = nn.Linear(D, H * (DSTATE // 2), bias=False)
        self.forget_proj = nn.Linear(D, H, bias=True)
        self.lambda_raw = nn.Parameter(torch.zeros(H))

    def forward(self, x, trace=None, tag="sisa"):
        bsz, seq_len, _ = x.shape
        dh = D // H
        q = self.W_Q(x).view(bsz, seq_len, H, dh).transpose(1, 2)
        k = self.W_K(x).view(bsz, seq_len, H, dh).transpose(1, 2)
        v = self.W_V(x).view(bsz, seq_len, H, dh).transpose(1, 2)
        b = self.W_B(x).view(bsz, seq_len, H, DSTATE).transpose(1, 2)
        c = self.W_C(x).view(bsz, seq_len, H, DSTATE).transpose(1, 2)
        a = self.alpha_proj(x).transpose(1, 2)                             # (B,H,L)
        theta = self.theta_proj(x).view(bsz, seq_len, H, DSTATE // 2).transpose(1, 2)
        g = torch.cumsum(-F.softplus(a), dim=2)
        phi = torch.cumsum(theta, dim=2)
        z = F.linear(x, self.forget_proj.weight, self.forget_proj.bias)
        fgate = torch.cumsum(F.logsigmoid(z).transpose(1, 2), dim=2)
        c = rotate_pairs(c, phi)
        b = rotate_pairs(b, phi)
        lam = F.softplus(self.lambda_raw)
        y = sisa_attention_ref(q, k, v, c, b, g, lam, fgate)
        out = self.W_O(y.transpose(1, 2).reshape(bsz, seq_len, D))
        if trace is not None:
            for name, val in (("q", q), ("k", k), ("v", v), ("c_rot", c), ("b_rot", b), ("g", g), ("phi", phi),
                              ("fgate", fgate), ("lam", lam), ("attn_y", y), ("attn_out", out)):
                trace[f"{tag}.{name}"] = val
        return out


class RMSNormParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        return rms_norm(x, self.weight, NORM_EPS)


class SwiGLUSisa(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_gate = nn.Linear(D, DFF, bias=False)
        self.w_up = nn.Linear(D, DFF, bias=False)
        self.w_down = nn.Linear(DFF, D, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SISABlockRef(nn.Module):
    """sisa.SISABlock: x + attn(attn_norm(x)); x + ffn(ffn_norm(x))."""

    def __init__(self):
        super().__init__()
        self.attn_norm = RMSNormParam()
        self.attn = SISAAttentionRef()
        self.ffn_norm = RMSNormParam()
        self.ffn = SwiGLUSisa()

    def forward(self, x, trace=None):
        xn = self.attn_norm(x)
        x = x + self.attn(xn, trace, "sisa")
        xf = self.ffn_norm(x)
        f = self.ffn(xf)
        if trace is not None:
            trace["sisa.attn_norm"] = xn
            trace["sisa.after_attn"] = x
            trace["sisa.ffn_norm"] = xf
            trace["sisa.ffn_out"] = f
        return x + f


class ShieldVRef(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v3_large
        self.cnn = mobilenet_v3_large(weights=None).features
        self.projection = nn.Linear(960, D, bias=False)
        self.time_embedding = nn.Parameter(torch.zeros(20, D))
        self.row_embedding = nn.Parameter(torch.zeros(4, D))
        self.col_embedding = nn.Parameter(torch.zeros(4, D))
        self.readout = nn.Parameter(torch.zeros(1, 1, D))
        self.gdn_a = GDNStageRef(repeats=1)
        self.gdn_b = GDNStageRef(repeats=2)
        self.sisa = SISABlockRef()
        self.final_norm = nn.RMSNorm(D, eps=NORM_EPS)
        self.head = nn.Sequential(nn.Linear(D, 128), nn.SiLU(), nn.Linear(128, 1))
        self.threshold = None

    @property
    def dtype(self):
        return self.projection.weight.dtype

    def cnn_map(self, frames):
        """Normalised [N,3,180,320] -> MobileNetV3 features [N,960,6,10]."""
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (3, 180, 320):
            raise ValueError(f"Expected [N,3,180,320], got {tuple(frames.shape)}")
        return self.cnn(frames)

    def tokens_from_map(self, fmap):
        features = F.adaptive_avg_pool2d(fmap, (4, 4))
        return self.projection(features.flatten(2).transpose(1, 2))

    def encode_frames(self, frames):
        """Normalised [N,3,180,320] -> tokens [N,16,384] before positions (ShieldV.encode_frames)."""
        return self.tokens_from_map(self.cnn_map(frames))

    def positions(self):
        return (self.time_embedding[:, None, None, :] + self.row_embedding[None, :, None, :]
                + self.col_embedding[None, None, :, :]).reshape(1, 20, 16, D)

    def classify_features(self, features, trace=None):
        if features.ndim != 4 or tuple(features.shape[1:]) != (20, 16, D):
            raise ValueError("Expected [B,20,16,384] features")
        batch = features.shape[0]
        h = (features + self.positions()).reshape(batch, 320, D)
        h = torch.cat((h, self.readout.expand(batch, -1, -1)), dim=1)
        if trace is not None:
            trace["h0"] = h
        h = self.gdn_a(h, trace, "gdn_a")
        if trace is not None:
            trace["after_gdn_a"] = h
        h = self.gdn_b(h, trace, "gdn_b")
        if trace is not None:
            trace["after_gdn_b"] = h
        h = self.sisa(h, trace)
        if trace is not None:
            trace["after_sisa"] = h
        z = self.final_norm(h[:, 320])
        hid = self.head[1](self.head[0](z))
        logit = self.head[2](hid).squeeze(-1)
        if trace is not None:
            trace["final_norm_row"] = z
            trace["head_hidden"] = hid
            trace["logit"] = logit
        return logit

    def forward(self, clips, trace=None):
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (20, 3, 180, 320):
            raise ValueError("Expected normalized clips [B,20,3,180,320]")
        batch = clips.shape[0]
        fmap = self.cnn_map(clips.reshape(-1, 3, 180, 320))
        tokens = self.tokens_from_map(fmap)
        if trace is not None:
            trace["cnn_map"] = fmap.reshape(batch, 20, 960, 6, 10)
            trace["tokens"] = tokens.reshape(batch, 20, 16, D)
        return self.classify_features(tokens.reshape(batch, 20, 16, D), trace)

    @torch.no_grad()
    def predict_uint8(self, raw, batch: int = 1):
        """uint8 clips [B,20,180,320,3] -> logits (numpy, model dtype), processed `batch` clips at a time."""
        raw = np.asarray(raw)
        out = []
        for i in range(0, raw.shape[0], batch):
            out.append(self(normalize(raw[i:i + batch], self.dtype)).numpy())
        return np.concatenate(out)


def _finish(model: ShieldVRef, dtype) -> ShieldVRef:
    total = sum(p.numel() for p in model.parameters())
    if total != PARAMETERS:
        raise RuntimeError(f"parameter count {total} != {PARAMETERS}")
    return model.to(dtype).eval().requires_grad_(False)


def load_ref(checkpoint=DEFAULT_CKPT, dtype=torch.float32) -> ShieldVRef:
    """Strict, weights_only load of a ShieldV training checkpoint into the CPU oracle."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ShieldVRef()
    model.load_state_dict(state["model"], strict=True)
    model.threshold = float(state["threshold"])
    return _finish(model, dtype)


def read_shldv(path=WEIGHTS) -> dict:
    """Parse the SHLDV001 export (tools/export_v.py) into {name: float32 array}."""
    out = {}
    with open(path, "rb") as f:
        if f.read(8) != b"SHLDV001":
            raise ValueError("not a SHLDV001 file")
        (count,) = struct.unpack("<I", f.read(4))
        for _ in range(count):
            (n,) = struct.unpack("<I", f.read(4))
            name = f.read(n).decode()
            (rank,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<" + "I" * rank, f.read(4 * rank))
            (elements,) = struct.unpack("<Q", f.read(8))
            out[name] = np.frombuffer(f.read(4 * elements), dtype="<f4").reshape(dims).copy()
        if f.read(1):
            raise ValueError("trailing bytes")
    return out


def load_ref_from_weights(path=WEIGHTS, dtype=torch.float32) -> ShieldVRef:
    """Build the oracle from the exported weight file (num_batches_tracked is not exported; it is unused in eval)."""
    tensors = read_shldv(path)
    model = ShieldVRef()
    sd = model.state_dict()
    missing = [k for k in sd if k not in tensors and not k.endswith("num_batches_tracked")]
    extra = [k for k in tensors if k not in sd]
    if missing or extra:
        raise RuntimeError(f"weight file mismatch: missing {missing[:5]}, extra {extra[:5]}")
    for k, v in sd.items():
        if k in tensors:
            if tuple(v.shape) != tensors[k].shape:
                raise RuntimeError(f"shape mismatch {k}")
            v.copy_(torch.from_numpy(tensors[k]))
    meta = json.loads(Path(str(path) + ".json").read_text())
    model.threshold = float(meta["threshold"])
    return _finish(model, dtype)


def threshold_from_weights_json(path=Path(str(WEIGHTS) + ".json")) -> float:
    return float(json.loads(Path(path).read_text())["threshold"])


def sigmoid64(logit: float) -> float:
    """Score in double precision, as the native contract requires."""
    x = float(logit)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
