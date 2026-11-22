"""M2: tokens to arithmetic.

Exact multiply-add (MAC) and memory-traffic counts per token for decoder-only
transformers in the Qwen3 layout (GQA attention with q/k norms, SwiGLU MLP, RMSNorm),
from the model's config.json. Then a check of the MAC formula against PyTorch's
FLOP counter on our own minimal implementation with random weights (no weights downloaded).

Usage:
  python -I m2_ops.py <configs dir> <M1 summary.csv> <out dir>

Conventions:
  - 1 MAC = one multiply plus one add. PyTorch's FlopCounterMode reports 2 FLOPs per MAC.
  - ctx = number of tokens attended to, including the current one.
  - Weights and KV cache stored in bf16 (2 bytes), as the configs specify.
  - Norms, activations, softmax and rotary embedding are counted separately as "elementwise"
    and are well under 1% of MACs at these sizes; they are reported, not hidden.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BYTES = 2  # bf16


def shapes(c: dict) -> dict:
    d, i, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    H, KV, hd, V = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"]
    per_layer_linear = d * H * hd + 2 * d * KV * hd + H * hd * d + 3 * d * i
    return dict(d=d, i=i, L=L, H=H, KV=KV, hd=hd, V=V, tied=c.get("tie_word_embeddings", False),
                per_layer_linear=per_layer_linear)


def macs_per_token(s: dict, ctx: int) -> dict:
    """MACs to process one token that attends to ctx tokens (decode step, or one prefill position)."""
    linear = s["L"] * s["per_layer_linear"]
    attention = s["L"] * 2 * s["H"] * s["hd"] * ctx          # q.k over ctx keys, then weights x values
    lm_head = s["d"] * s["V"]
    return {"linear": linear, "attention": attention, "lm_head": lm_head, "total": linear + attention + lm_head}


def bytes_per_token(s: dict, ctx: int, batch: int = 1) -> dict:
    """Memory traffic for one decode step. Weights are read once per step and shared by the batch."""
    weights = (s["L"] * s["per_layer_linear"] + s["d"] * s["V"]) * BYTES
    kv_read = 2 * s["L"] * s["KV"] * s["hd"] * ctx * BYTES
    kv_write = 2 * s["L"] * s["KV"] * s["hd"] * BYTES
    return {"weights_per_token": weights / batch, "kv_read": kv_read, "kv_write": kv_write,
            "total": weights / batch + kv_read + kv_write}


def sentence_prefill_macs(s: dict, n: int) -> int:
    """MACs to read an n-token input from scratch (position p attends to p tokens)."""
    linear_and_head = n * (s["L"] * s["per_layer_linear"] + s["d"] * s["V"])
    attention = s["L"] * 2 * s["H"] * s["hd"] * n * (n + 1) // 2
    return linear_and_head + attention


def torch_check(cfg: dict, ctx: int) -> dict:
    """Count FLOPs of one decode step on a minimal Qwen3-layout model with random weights."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    torch.manual_seed(0)
    s = shapes(cfg)
    d, H, KV, hd, I, V, L = s["d"], s["H"], s["KV"], s["hd"], s["i"], s["V"], s["L"]

    def lin(a, b):
        return torch.nn.Linear(a, b, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v, self.o = lin(d, H * hd), lin(d, KV * hd), lin(d, KV * hd), lin(H * hd, d)
            self.g, self.u, self.dn = lin(d, I), lin(d, I), lin(I, d)
            self.n1, self.n2 = torch.nn.RMSNorm(d), torch.nn.RMSNorm(d)
            self.qn, self.kn = torch.nn.RMSNorm(hd), torch.nn.RMSNorm(hd)

        def forward(self, x, kc, vc):
            h = self.n1(x)
            q = self.qn(self.q(h).view(1, 1, H, hd)).transpose(1, 2)
            k = self.kn(self.k(h).view(1, 1, KV, hd)).transpose(1, 2)
            v = self.v(h).view(1, 1, KV, hd).transpose(1, 2)
            k, v = torch.cat([kc, k], 2), torch.cat([vc, v], 2)
            k, v = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
            att = torch.softmax(q @ k.transpose(-1, -2) / hd ** 0.5, -1) @ v
            x = x + self.o(att.transpose(1, 2).reshape(1, 1, H * hd))
            h = self.n2(x)
            return x + self.dn(torch.nn.functional.silu(self.g(h)) * self.u(h))

    layers = torch.nn.ModuleList(Layer() for _ in range(L))
    head = lin(d, V)
    x = torch.randn(1, 1, d)
    caches = [(torch.randn(1, KV, ctx - 1, hd), torch.randn(1, KV, ctx - 1, hd)) for _ in range(L)]
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        for layer, (kc, vc) in zip(layers, caches):
            x = layer(x, kc, vc)
        head(x)
    counted_macs = fc.get_total_flops() // 2
    return {"ctx": ctx, "counted_macs": counted_macs, "formula_macs": macs_per_token(s, ctx)["total"]}


def main() -> None:
    cfg_dir, m1_summary, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    models = {p.stem: json.loads(p.read_text()) for p in sorted(cfg_dir.glob("Qwen3-*.json"))}
    result = {"models": {}, "checks": []}

    for name, cfg in models.items():
        s = shapes(cfg)
        params_linear = s["L"] * s["per_layer_linear"] + s["d"] * s["V"]
        rows = []
        for ctx in (1, 128, 512, 2048, 8192, 32768):
            m, b = macs_per_token(s, ctx), bytes_per_token(s, ctx)
            rows.append({"ctx": ctx, **{f"mac_{k}": v for k, v in m.items()},
                         **{f"byte_{k}": v for k, v in b.items()},
                         "attention_share": m["attention"] / m["total"],
                         "kv_share_of_bytes_batch1": b["kv_read"] / b["total"],
                         "kv_share_of_bytes_batch64": bytes_per_token(s, ctx, 64)["kv_read"]
                         / bytes_per_token(s, ctx, 64)["total"]})
        result["models"][name] = {"shapes": s, "linear_params": params_linear, "per_ctx": rows}

    # Per-language cost of reading one FLORES sentence with Qwen3-8B (Qwen3 tokenizer counts from M1).
    s8 = shapes(models["Qwen3-8B"])
    langs = []
    for r in csv.DictReader(open(m1_summary, encoding="utf-8")):
        if r["tokenizer"] != "qwen3":
            continue
        n = float(r["tokens_per_sentence"])
        langs.append({"lang": r["lang"], "tokens_per_sentence": n,
                      "prefill_gmacs": sentence_prefill_macs(s8, round(n)) / 1e9})
    eng = next(x for x in langs if x["lang"] == "eng_Latn")["prefill_gmacs"]
    for x in langs:
        x["ratio_to_eng"] = x["prefill_gmacs"] / eng
    result["qwen3_8b_sentence_prefill"] = sorted(langs, key=lambda x: -x["ratio_to_eng"])

    for ctx in (1, 64, 300):
        result["checks"].append(torch_check(models["Qwen3-0.6B"], ctx))

    (out / "m2_ops.json").write_text(json.dumps(result, indent=2))
    for name, m in result["models"].items():
        r1, r8k = m["per_ctx"][0], m["per_ctx"][4]
        print(f"{name:11s} linear params {m['linear_params']/1e9:6.3f} B | MACs/token ctx1 {r1['mac_total']/1e9:6.2f} G,"
              f" ctx8192 {r8k['mac_total']/1e9:6.2f} G (attention {r8k['attention_share']:.1%}) |"
              f" bytes/token ctx8192 batch1 {r8k['byte_total']/1e9:5.2f} GB, KV share b1 {r8k['kv_share_of_bytes_batch1']:.1%}"
              f" b64 {r8k['kv_share_of_bytes_batch64']:.1%}")
    for c in result["checks"]:
        print(f"check Qwen3-0.6B ctx {c['ctx']:4d}: counted {c['counted_macs']:,} formula {c['formula_macs']:,}"
              f" diff {c['counted_macs']-c['formula_macs']:+,} ({(c['counted_macs']-c['formula_macs'])/c['formula_macs']:+.4%})")
    top = result["qwen3_8b_sentence_prefill"]
    print("Qwen3-8B, read one sentence: eng", round(eng, 1), "GMAC; top:",
          [(x["lang"], round(x["ratio_to_eng"], 2)) for x in top[:5]])


if __name__ == "__main__":
    main()
