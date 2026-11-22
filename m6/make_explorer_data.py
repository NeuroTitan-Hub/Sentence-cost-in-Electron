"""Data for the post's interactive electron explorer, computed only from files in this repo.
Run from the repo root:  python -I m6/make_explorer_data.py  ->  m6/explorer.json

Per language (Qwen3 tokenizer, FLORES-200 mean sentence length, rounded to whole tokens):
  multiply-adds to read the sentence  M(n) = n * M_lin + 2*l*H*h * n(n+1)/2      (derived, post section 2)
Per supply (sky130, typical corner, 27 C, 8-bit multiply-add, lower bound):
  energy per multiply-add, one ideal copy and with chip-to-chip variation      (simulated, M5 logic Monte Carlo)
The page computes  E = M * E_MAC  and  N_e = E / (q V)  in the browser and shows the formula.

Names: taken from Deng et al.'s Table 5 (m4/deng2026_table5.csv), which covers 117 of the 204 FLORES
languages. The other 87 are shown by their FLORES code; we have no offline name list we trust and
will not type names from memory.
"""
import csv
import json
from pathlib import Path

m2 = json.load(open("m2/m2_ops.json"))["models"]["Qwen3-8B"]["shapes"]
M_LIN = m2["L"] * (m2["d"] * m2["H"] * m2["hd"] + 2 * m2["d"] * m2["KV"] * m2["hd"]
                   + m2["H"] * m2["hd"] * m2["d"] + 3 * m2["d"] * m2["i"]) + m2["d"] * m2["V"]
ATT = 2 * m2["L"] * m2["H"] * m2["hd"]
assert M_LIN == 7568097280 and ATT == 294912

names = {r["code"]: r["language"] for r in csv.DictReader(open("m4/deng2026_table5.csv", encoding="utf-8"))}
langs = []
for r in csv.DictReader(open("m1/summary.csv", encoding="utf-8")):
    if r["tokenizer"] != "qwen3":
        continue
    tps = float(r["tokens_per_sentence"])
    n = round(tps)
    langs.append({"code": r["lang"], "name": names.get(r["lang"]), "tokens": round(tps, 1), "n": n,
                  "macs": n * M_LIN + ATT * n * (n + 1) // 2, "ratio": float(r["ratio_to_eng"]),
                  "lo": float(r["ratio_lo95"]), "hi": float(r["ratio_hi95"])})
langs.sort(key=lambda x: (x["name"] is None, x["name"] or x["code"]))

lg = {(r["corner"], float(r["vdd"])): r for r in csv.DictReader(open("m5/logic_mc_summary.csv", encoding="utf-8"))}
# clock period: one ideal copy = 30 carry delays of the nominal adder; with variation = the 3-sigma guarded clock
supplies = [{"v": v, "ideal": float(lg[("tt", v)]["e_mac_nominal"]), "variation": float(lg[("tt", v)]["e_mac_guarded"]),
             "cycle_ideal": 30 * float(lg[("tt", v)]["tpd_nominal"]), "cycle_variation": float(lg[("tt", v)]["cycle_guarded_s"])}
            for v in sorted({v for c, v in lg if c == "tt"})]

# ---- whole request on a GPU server (M4 chain): N input sentences, 5 output sentences, alone (b=1) or 64 users
import sys
sys.path.insert(0, "m4")
import m4_chain as CH  # noqa: E402
cfg = json.load(open("m4/chain_config.json"))
chain_rows = {r["lang"]: r for r in csv.DictReader(open("m4/chain_per_language.csv", encoding="utf-8"))}
configs = [f"N{N}_b{b}" for N in cfg["n_input"] for b in cfg["batch"]]
em, eb = CH.E_MAC[CH.REF[0]], CH.E_BIT[CH.REF[1]]
eng_s = {}
for N in cfg["n_input"]:
    for b in cfg["batch"]:
        e = CH.task(cfg["shapes"], cfg["n_eng"], N, b, em, eb)
        P, O = e["P"], e["O"]
        quad = (cfg["shapes"]["attn_macs_per_ctx"] * (P * P / 2 + O * P + O * O / 2) * em
                + cfg["shapes"]["kv_bytes_per_tok"] * (O * P + O * O / 2) * 8 * eb)
        eng_s[f"N{N}_b{b}"] = round(quad / e["e_total"], 4)
for L in langs:
    r = chain_rows[L["code"]]
    L["chain"] = {k: [float(r[f"{k}_ref_J"]), float(r[f"{k}_ratio"]), float(r[f"{k}_ratio_lo"]), float(r[f"{k}_ratio_hi"]),
                      float(r[f"{k}_share_arith"]), float(r[f"{k}_share_weights"]), float(r[f"{k}_share_kv"])] for k in configs}
chain_meta = {"configs": configs, "fields": ["J", "ratio", "lo", "hi", "share_arith", "share_weights", "share_kv"],
              "english_quadratic_share": eng_s, "out_sentences": cfg["out_sentences"],
              "e_mac": CH.REF[0], "e_bit": CH.REF[1],
              "range_over": "3 multiply-add energies x 3 memory energies x the token count's 95% interval"}

out = {"q": 1.602176634e-19, "m_lin": M_LIN, "att_per_ctx": ATT, "electrons_per_dot": 1e16,
       "sources": {"tokens": "m1/summary.csv", "macs": "m2 shapes", "energy": "m5/logic_mc_summary.csv (tt)",
                   "names": "m4/deng2026_table5.csv"},
       "supplies": supplies, "chain": chain_meta, "languages": langs}
Path("m6/explorer.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
eng = next(x for x in langs if x["code"] == "eng_Latn")
s18 = next(s for s in supplies if s["v"] == 1.8)
print(len(langs), "languages,", sum(x["name"] is not None for x in langs), "named;", len(supplies), "supplies")
print("check English 1.8 V electrons:", f'{eng["macs"] * s18["ideal"] / (out["q"] * 1.8):.3g}')
