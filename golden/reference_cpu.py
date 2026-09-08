"""fp32 reference of MiniCPM5-2B (Llama architecture) in plain torch, read straight from the bf16
safetensors checkpoint. Used for (1) calibration statistics, (2) the accuracy baseline the integer golden
model is gated against. Mirrors HF's LlamaForCausalLM numerics in fp32 (validated by golden/crosscheck_hf.py).

  uv run python golden/reference_cpu.py --prompt "..." --max-new 16
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from golden.data import CHECKPOINT, CONFIG, EOS_IDS, chat_ids, load_tokenizer  # noqa: E402

torch.set_grad_enabled(False)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class LlamaRef:
    """Weights stay bf16 in the mmapped checkpoint; each layer is converted to fp32 when used."""

    def __init__(self, path=CHECKPOINT, n_layers: int | None = None):
        from safetensors import safe_open
        self.cfg = json.load(open(CONFIG))
        c = self.cfg
        self.D, self.H, self.HKV, self.HD, self.FF = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["intermediate_size"]
        self.L = n_layers or c["num_hidden_layers"]
        self.eps, self.theta = c["rms_norm_eps"], c["rope_theta"]
        self.f = safe_open(str(path), framework="pt")
        inv = 1.0 / (self.theta ** (torch.arange(0, self.HD, 2, dtype=torch.int64).float() / self.HD))
        self.inv_freq = inv.float()
        self.stats = None

    def w(self, name) -> torch.Tensor:
        return self.f.get_tensor(name).float()

    def rms(self, x, weight):
        var = x.pow(2).mean(-1, keepdim=True)
        return weight * (x * torch.rsqrt(var + self.eps))

    def rope_cs(self, positions):
        pos = torch.as_tensor(positions, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]            # [T, 64] fp32
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()                              # [T, 128]

    # ---- stats helpers (calibration)
    def _mx(self, key, t):
        if self.stats is not None:
            v = float(t.abs().max())
            self.stats["max"][key] = max(self.stats["max"].get(key, 0.0), v)

    def _chan(self, key, t):
        if self.stats is not None:
            v = t.abs().amax(dim=0).numpy()
            old = self.stats["chan"].get(key)
            self.stats["chan"][key] = v if old is None else np.maximum(old, v)

    def _head(self, key, t):   # t [T, heads, hd]
        if self.stats is not None:
            v = t.abs().amax(dim=(0, 2)).numpy()
            old = self.stats["head"].get(key)
            self.stats["head"][key] = v if old is None else np.maximum(old, v)

    def forward(self, ids, kv=None, pos0: int = 0):
        """ids [T] -> logits [T, V] fp32. kv: optional list of (k, v) per layer to append to (fp32 tensors)."""
        ids = torch.as_tensor(list(ids), dtype=torch.long)
        T = ids.shape[0]
        x = self.w("model.embed_tokens.weight")[ids]            # [T, D]
        positions = torch.arange(pos0, pos0 + T)
        cos, sin = self.rope_cs(positions)
        for l in range(self.L):
            p = f"model.layers.{l}"
            self._mx(f"x_in.{l}", x)
            h = self.rms(x, self.w(f"{p}.input_layernorm.weight"))
            self._chan(f"norm1.{l}", h)
            q = (h @ self.w(f"{p}.self_attn.q_proj.weight").T).view(T, self.H, self.HD)
            k = (h @ self.w(f"{p}.self_attn.k_proj.weight").T).view(T, self.HKV, self.HD)
            v = (h @ self.w(f"{p}.self_attn.v_proj.weight").T).view(T, self.HKV, self.HD)
            self._head(f"q_pre.{l}", q); self._head(f"k_pre.{l}", k)
            q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
            k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
            self._head(f"q.{l}", q); self._head(f"k.{l}", k); self._mx(f"v.{l}", v)
            if kv is not None:
                if kv[l] is None:
                    kv[l] = (k, v)
                else:
                    kv[l] = (torch.cat([kv[l][0], k]), torch.cat([kv[l][1], v]))
                kf, vf = kv[l]
            else:
                kf, vf = k, v
            S = kf.shape[0]
            rep = self.H // self.HKV
            kr = kf.repeat_interleave(rep, dim=1)                  # [S, H, HD]
            vr = vf.repeat_interleave(rep, dim=1)
            scores = torch.einsum("thd,shd->hts", q, kr) / math.sqrt(self.HD)
            mask = torch.arange(S)[None, :] > (pos0 + torch.arange(T))[:, None]
            scores = scores.masked_fill(mask[None], float("-inf"))
            pr = torch.softmax(scores, dim=-1)
            att = torch.einsum("hts,shd->thd", pr, vr).reshape(T, self.D)
            o = att @ self.w(f"{p}.self_attn.o_proj.weight").T
            self._mx(f"o.{l}", o)
            x = x + o
            self._mx(f"x_mid.{l}", x)
            h = self.rms(x, self.w(f"{p}.post_attention_layernorm.weight"))
            self._chan(f"norm2.{l}", h)
            g = h @ self.w(f"{p}.mlp.gate_proj.weight").T
            u = h @ self.w(f"{p}.mlp.up_proj.weight").T
            self._mx(f"gate.{l}", g); self._mx(f"up.{l}", u)
            hh = torch.nn.functional.silu(g) * u
            self._mx(f"h.{l}", hh)
            dn = hh @ self.w(f"{p}.mlp.down_proj.weight").T
            self._mx(f"down.{l}", dn)
            x = x + dn
        self._mx(f"x_in.{self.L}", x)
        h = self.rms(x, self.w("model.norm.weight"))
        self._chan("norm_f", h)
        return h @ self.w("lm_head.weight").T

    def generate(self, ids, max_new: int, eos=EOS_IDS, verbose=False):
        kv = [None] * self.L
        logits = self.forward(ids, kv, 0)
        nxt = int(torch.argmax(logits[-1]))
        out = []
        pos = len(ids)
        while nxt not in eos and len(out) < max_new:
            out.append(nxt)
            if verbose:
                print(nxt, end=" ", flush=True)
            logits = self.forward([nxt], kv, pos)
            pos += 1
            nxt = int(torch.argmax(logits[-1]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one word.")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args()
    tok = load_tokenizer()
    ids = chat_ids(tok, a.prompt)
    m = LlamaRef(n_layers=a.layers)
    out = m.generate(ids, a.max_new, verbose=True)
    print("\n", repr(tok.decode(out)))


if __name__ == "__main__":
    main()
