"""M4 test 1: can counted memory traffic and arithmetic predict Deng et al.'s measured per-language energy per token?

Deng et al. (arXiv 2606.21869v3, Table 5) measured GPU energy per output token for Qwen3-8B on Belebele,
122 languages, one NVIDIA L40S, vLLM, maximum batch size 256, steady state. Their explanation is in words:
long outputs fill the KV cache, which forces smaller batches and more memory traffic per token.

We turn that sentence into numbers, with NO parameter fitted to their energy measurements:

  inputs per language L
    O_L  mean output tokens per response = their output tokens / 900 Belebele questions   (from their table)
    P_L  prompt tokens = P_eng x our M1 Qwen3 token ratio for L (FLORES-200)               (P_eng swept)
  one decode step of the server, b sequences in flight
    c    mean context a sequence attends to = P_L + O_L / 2   (each sequence spends one step per output token)
    b    = min(256, KV_budget / (kv_bytes_per_token x c))      (vLLM can only keep what fits in GPU memory)
    bytes  = weights + b x kv_bytes_per_token x c              (weights read once per step, KV read per sequence)
    flops  = 2 x b x (MACs/token at context c)  +  prefill of the prompts that arrive (b / O_L new prompts per step)
    time   = max(bytes / bandwidth, flops / peak)              (roofline; whichever unit is the bottleneck)
  per output token
    GPU-time model:  E ~ power x time / b   -> ratio to English does not depend on power
    physics floor:   E = (bytes x 8 x e_bit + MACs x e_mac) / b, with e_bit for GDDR6 (M3b sources) and our M3a e_mac

Known simplifications (all push the same way unless noted):
  - c uses mean output length. Real outputs vary; because long responses occupy the server for more steps, the
    true token-weighted context is larger than P + mean(O)/2 (Jensen). So for languages with long, variable outputs
    we expect to UNDER-predict.
  - Ideal bandwidth and peak (no efficiency factor). This changes absolute numbers, not ratios, unless the
    bottleneck flips between compute and memory.
  - Belebele prompts are built on FLORES passages; we scale an assumed English prompt length by our FLORES token
    ratio. P_eng was first swept; it is now measured at 220 tokens (m4_belebele_prompt.py, Deng et al.'s Appendix B
    template on English Belebele, Qwen3 tokenizer, with chat wrapper; 212 without). 220 is the baseline; the sweep stays.

Usage: python -I m4_deng_test.py <m1 summary.csv> <Qwen3-8B config.json> <deng2026_table5.csv> <out dir>
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

# NVIDIA L40S, from nvidia.com/en-eu/data-center/l40s (opened 2026-11-19)
L40S = {"mem_bytes": 48e9, "bw": 864e9, "peak_bf16": 362.05e12, "power_w": 350.0}
BYTES = 2                      # bf16 weights and KV cache
BATCH_CAP = 256                # Deng et al. section 4.1
QUESTIONS = 900                # Belebele questions per language
# Swept, because we do not know them:
P_ENG = [50, 150, 220, 300, 500]   # English prompt tokens (passage + question + 4 options + instruction)
BASE_P = 220                       # measured: belebele_prompt.json, chat-template mean 219.9
KV_BUDGET = [20e9, 23e9, 26e9] # bytes of GPU memory left for KV cache after weights and runtime overhead
E_BIT = {"GDDR6 (vendor, 7.5 pJ/bit)": 7.5e-12, "GDDR5 (model, 14 pJ/bit)": 14.0e-12}


def model_shapes(c: dict) -> dict:
    d, i, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    H, KV, hd, V = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"]
    linear = L * (d * H * hd + 2 * d * KV * hd + H * hd * d + 3 * d * i) + d * V   # same as M2
    return {"linear_macs": linear, "attn_macs_per_ctx": L * 2 * H * hd,
            "weight_bytes": linear * BYTES, "kv_bytes_per_tok": 2 * L * KV * hd * BYTES}


def step(s: dict, P: float, O: float, kv_budget: float, e_bit: float, e_mac: float) -> dict:
    c = P + O / 2
    b = min(BATCH_CAP, kv_budget / (s["kv_bytes_per_tok"] * c))
    bytes_ = s["weight_bytes"] + b * s["kv_bytes_per_tok"] * c
    macs_decode = b * (s["linear_macs"] + s["attn_macs_per_ctx"] * c)
    new_prompts = b / O
    macs_prefill = new_prompts * (P * s["linear_macs"] + s["attn_macs_per_ctx"] * P * (P + 1) / 2)
    macs = macs_decode + macs_prefill
    t_mem, t_cmp = bytes_ / L40S["bw"], 2 * macs / L40S["peak_bf16"]
    t = max(t_mem, t_cmp)
    return {"ctx": c, "batch": b, "t_per_tok": t / b, "mem_bound": t_mem >= t_cmp,
            "floor_j_per_tok": (bytes_ * 8 * e_bit + macs * e_mac) / b}


def spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation with average ranks for ties (their J/token has 2 decimals, so many ties)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    return pearson(ranks(x), ranks(y))


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def main() -> None:
    m1, cfg, deng, out = (Path(a) for a in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    s = model_shapes(json.loads(cfg.read_text()))
    ratio = {r["lang"]: float(r["ratio_to_eng"]) for r in csv.DictReader(open(m1, encoding="utf-8"))
             if r["tokenizer"] == "qwen3"}
    D = list(csv.DictReader(open(deng, encoding="utf-8")))
    for r in D:
        r["O"] = float(r["output_tokens"]) / QUESTIONS
        r["meas"] = float(r["total_energy_j"]) / float(r["output_tokens"])   # unrounded J/token
    eng = next(r for r in D if r["code"] == "eng_Latn")
    both = [r for r in D if r["code"] in ratio]
    missing = [r["code"] for r in D if r["code"] not in ratio]
    e_mac = 1.25e-12   # M3a corrected (with junctions), 8-bit MAC at 1.8 V tt (lower bound); was 1.07 before the fix

    runs, rows = [], []
    for p_eng in P_ENG:
        for kvb in KV_BUDGET:
            for e_name, e_bit in E_BIT.items():
                def pred(r):
                    return step(s, p_eng * ratio[r["code"]], r["O"], kvb, e_bit, e_mac)
                pe = pred(eng)
                xs_t, xs_f, ys = [], [], []
                for r in both:
                    p = pred(r)
                    xs_t.append(p["t_per_tok"] / pe["t_per_tok"])
                    xs_f.append(p["floor_j_per_tok"] / pe["floor_j_per_tok"])
                    ys.append(r["meas"] / eng["meas"])
                    if p_eng == BASE_P and kvb == 23e9 and e_name.startswith("GDDR6"):
                        rows.append({"code": r["code"], "language": r["language"], "out_tokens_per_resp": round(r["O"], 1),
                                     "prompt_tokens": round(p_eng * ratio[r["code"]]), "ctx": round(p["ctx"]),
                                     "batch": round(p["batch"], 1), "mem_bound": p["mem_bound"],
                                     "pred_ratio_time": round(xs_t[-1], 3), "pred_ratio_floor": round(xs_f[-1], 3),
                                     "meas_ratio": round(ys[-1], 3)})
                lt = [math.log(a / b) for a, b in zip(xs_t, ys)]
                lf = [math.log(a / b) for a, b in zip(xs_f, ys)]
                runs.append({"p_eng": p_eng, "kv_budget_gb": kvb / 1e9, "e_bit": e_name,
                             "spearman_time": spearman(xs_t, ys), "pearson_log_time": pearson([math.log(v) for v in xs_t], [math.log(v) for v in ys]),
                             "median_abs_err_time": math.exp(sorted(abs(v) for v in lt)[len(lt) // 2]),
                             "spearman_floor": spearman(xs_f, ys),
                             "median_abs_err_floor": math.exp(sorted(abs(v) for v in lf)[len(lf) // 2]),
                             "max_pred_time": max(xs_t), "max_pred_floor": max(xs_f), "max_meas": max(ys),
                             "implied_power_w_eng": eng["meas"] / pe["t_per_tok"]})
    ys0 = [r["meas"] / eng["meas"] for r in both]
    baselines = {"output length alone": [r["O"] for r in both],
                 "prompt length alone": [ratio[r["code"]] for r in both],
                 f"mean context (P_eng {BASE_P})": [BASE_P * ratio[r["code"]] + r["O"] / 2 for r in both]}
    base_res = {k: spearman(v, ys0) for k, v in baselines.items()}
    with open(out / "per_language.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["meas_ratio"]))
    (out / "deng_test.json").write_text(json.dumps({"n_languages": len(both), "not_in_flores_m1": missing,
                                                    "shapes": s, "gpu": L40S, "runs": runs, "baselines_spearman": base_res}, indent=2))
    print(f"languages compared: {len(both)} (missing from M1: {missing})")
    print("P_eng  KV GB  e_bit                      | time model: spearman  logR  med|err|  max pred | floor: spearman med|err| max | max meas | implied W")
    for r in runs:
        print(f"{r['p_eng']:5d} {r['kv_budget_gb']:5.0f}  {r['e_bit']:26s} | {r['spearman_time']:6.3f} {r['pearson_log_time']:6.3f} {r['median_abs_err_time']:6.2f}x"
              f" {r['max_pred_time']:7.2f}x | {r['spearman_floor']:6.3f} {r['median_abs_err_floor']:6.2f}x {r['max_pred_floor']:6.2f}x | {r['max_meas']:6.2f}x | {r['implied_power_w_eng']:6.0f}")
    print(f"\nbaseline (P_eng {BASE_P}, KV 23 GB), top 10 by measured ratio:")
    for r in sorted(rows, key=lambda r: -r["meas_ratio"])[:10]:
        print(f"  {r['code']:9s} out {r['out_tokens_per_resp']:7.0f} ctx {r['ctx']:6d} batch {r['batch']:6.1f}  pred {r['pred_ratio_time']:5.2f}x  floor {r['pred_ratio_floor']:5.2f}x  measured {r['meas_ratio']:5.2f}x")


if __name__ == "__main__":
    main()
