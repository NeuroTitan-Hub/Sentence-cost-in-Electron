"""M4 chain: energy for the same content in every FLORES-200 language, from counted parts only.

This is the "physics floor" view: energy of the arithmetic plus energy of moving bits to and from off-chip memory.
It leaves out idle/static power and everything else a real GPU spends while waiting (M4 test 1 shows that part is large;
see README). Every input is either counted (M1, M2), simulated (M3a, M3b) or read from an opened source (m3b/sources.md).

Task (same meaning in every language): read a text of N FLORES sentences, then write an answer of 5 sentences.
  tokens per sentence n_L from M1 (Qwen3 tokenizer, FLORES-200 devtest, 1,012 sentences; 95% CI from M1's bootstrap)
  prefill: P = N * n_L tokens, all positions in one pass (weights read once, shared by the batch)
  decode:  O = 5 * n_L tokens, one step each; each step reads weights / batch and the whole KV cache so far

Per step the energy is  MACs * e_mac  +  bytes * 8 * e_bit.
Model: Qwen3-8B (counts from M2's formula, which matched PyTorch's FLOP counter exactly).

Usage: python -I m4_chain.py <m1 summary.csv> <Qwen3-8B config.json> <out dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BYTES = 2  # bf16 weights and KV cache, as the config specifies
OUT_SENTENCES = 5
N_INPUT = [1, 20, 200]          # a sentence, a page, a long document or chat history
BATCH = [1, 64]                 # one user alone vs a busy server
E_MAC = {                       # energy per multiply-add
    "sky130 int8 1.8V (M3a, lower bound)": 1.25e-12,
    "45nm int8 (Horowitz 2014: 0.2 + 0.03 pJ)": 0.23e-12,
    "45nm fp16 (Horowitz 2014: 1.1 + 0.4 pJ)": 1.5e-12,
}
E_BIT = {                       # off-chip energy per bit (m3b/sources.md)
    "HBM2 3.95 pJ (O'Connor 2017)": 3.95e-12,
    "GDDR6 7.5 pJ (Micron, vendor)": 7.5e-12,
    "DRAM 30 pJ (Horowitz 2014, 20-41 midpoint)": 30e-12,
}
REF = ("45nm fp16 (Horowitz 2014: 1.1 + 0.4 pJ)", "GDDR6 7.5 pJ (Micron, vendor)")


def shapes(c: dict) -> dict:
    d, i, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    H, KV, hd, V = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"]
    linear = L * (d * H * hd + 2 * d * KV * hd + H * hd * d + 3 * d * i) + d * V
    return {"linear_macs": linear, "attn_macs_per_ctx": L * 2 * H * hd,
            "weight_bytes": linear * BYTES, "kv_bytes_per_tok": 2 * L * KV * hd * BYTES}


def task(s: dict, n: float, N: int, batch: int, e_mac: float, e_bit: float) -> dict:
    """Energy of one request (its share of the batch) for N input sentences and 5 output sentences of n tokens each."""
    P, O = round(N * n), round(OUT_SENTENCES * n)
    kv = s["kv_bytes_per_tok"]
    # prefill: every position p (1..P) attends to p tokens
    pre_macs = P * s["linear_macs"] + s["attn_macs_per_ctx"] * P * (P + 1) / 2
    pre_bytes = s["weight_bytes"] / batch + kv * P                     # weights once, KV written once
    # decode: step k (0..O-1) attends to P + k + 1 tokens and reads the KV of the P + k tokens before it
    dec_macs = O * s["linear_macs"] + s["attn_macs_per_ctx"] * (O * P + O * (O + 1) / 2)
    dec_w = O * s["weight_bytes"] / batch
    dec_kv = kv * (O * P + O * (O - 1) / 2) + kv * O                  # reads + writes
    macs = pre_macs + dec_macs
    e_arith = macs * e_mac
    e_weights = (s["weight_bytes"] / batch + dec_w) * 8 * e_bit
    e_kv = (kv * P + dec_kv) * 8 * e_bit
    return {"P": P, "O": O, "macs": macs, "e_arith": e_arith, "e_weights": e_weights, "e_kv": e_kv,
            "e_total": e_arith + e_weights + e_kv}


def main() -> None:
    m1, cfg, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    s = shapes(json.loads(cfg.read_text()))
    L = [r for r in csv.DictReader(open(m1, encoding="utf-8")) if r["tokenizer"] == "qwen3"]
    eng = next(r for r in L if r["lang"] == "eng_Latn")
    n_eng = float(eng["tokens_per_sentence"])

    rows = []
    for r in L:
        n = float(r["tokens_per_sentence"])
        n_lo, n_hi = n_eng * float(r["ratio_lo95"]), n_eng * float(r["ratio_hi95"])
        row = {"lang": r["lang"], "tokens_per_sentence": n, "token_ratio": round(n / n_eng, 3)}
        for N in N_INPUT:
            for b in BATCH:
                ref = task(s, n, N, b, E_MAC[REF[0]], E_BIT[REF[1]])
                ref_e = task(s, n_eng, N, b, E_MAC[REF[0]], E_BIT[REF[1]])
                # ratio range over every memory/arithmetic choice and the token-count CI
                ratios = []
                for em in E_MAC.values():
                    for eb in E_BIT.values():
                        base = task(s, n_eng, N, b, em, eb)["e_total"]
                        for nn in (n_lo, n_hi):
                            ratios.append(task(s, nn, N, b, em, eb)["e_total"] / base)
                k = f"N{N}_b{b}"
                row[f"{k}_ref_J"] = round(ref["e_total"], 4)
                row[f"{k}_ratio"] = round(ref["e_total"] / ref_e["e_total"], 3)
                row[f"{k}_ratio_lo"] = round(min(ratios), 3)
                row[f"{k}_ratio_hi"] = round(max(ratios), 3)
                row[f"{k}_share_arith"] = round(ref["e_arith"] / ref["e_total"], 3)
                row[f"{k}_share_weights"] = round(ref["e_weights"] / ref["e_total"], 3)
                row[f"{k}_share_kv"] = round(ref["e_kv"] / ref["e_total"], 3)
        rows.append(row)
    rows.sort(key=lambda r: -r["token_ratio"])
    with open(out / "chain_per_language.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "chain_config.json").write_text(json.dumps({"model": "Qwen3-8B", "shapes": s, "e_mac": E_MAC, "e_bit": E_BIT,
                                                       "reference": REF, "out_sentences": OUT_SENTENCES,
                                                       "n_input": N_INPUT, "batch": BATCH, "n_eng": n_eng}, indent=2))

    show = ["eng_Latn", "fra_Latn", "zho_Hans", "hin_Deva", "ben_Beng", "ory_Orya", "pbt_Arab", "amh_Ethi", "tir_Ethi", "shn_Mymr"]
    by = {r["lang"]: r for r in rows}
    print(f"reference: {REF[0]} + {REF[1]};  English {n_eng:.1f} tokens/sentence")
    for N in N_INPUT:
        for b in BATCH:
            k = f"N{N}_b{b}"
            e = by["eng_Latn"]
            print(f"\n== input {N} sentence(s), batch {b}: English {e[k+'_ref_J']:.3f} J  "
                  f"(arith {e[k+'_share_arith']:.0%}, weights {e[k+'_share_weights']:.0%}, KV {e[k+'_share_kv']:.0%})")
            for c in show:
                x = by.get(c)
                if x:
                    print(f"  {c:9s} tokens x{x['token_ratio']:5.2f}  energy x{x[k+'_ratio']:6.2f}  [{x[k+'_ratio_lo']:.2f}, {x[k+'_ratio_hi']:.2f}]"
                          f"  KV share {x[k+'_share_kv']:.0%}")
            top = rows[0]
            print(f"  max: {top['lang']} tokens x{top['token_ratio']:.2f} energy x{top[k+'_ratio']:.2f}")


if __name__ == "__main__":
    main()
