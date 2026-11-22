"""Number check: recompute every number in the post from the files in this repo.
Run from the repo root:  python -I m6/popper_check.py
"""
import csv
import json
import math
from statistics import NormalDist


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


out = []


def show(label, value):
    out.append(f"{label:58s} {value}")


# --- M1 tokens
m1 = {(r["tokenizer"], r["lang"]): r for r in rows("m1/summary.csv")}
for L in ("eng_Latn", "ory_Orya", "shn_Mymr"):
    r = m1[("qwen3", L)]
    show(f"qwen3 {L} tok/sent, ratio [lo,hi]",
         f'{float(r["tokens_per_sentence"]):.1f}  {float(r["ratio_to_eng"]):.2f} '
         f'[{float(r["ratio_lo95"]):.2f}, {float(r["ratio_hi95"]):.2f}]')
q = sorted(((float(r["ratio_to_eng"]), l) for (t, l), r in m1.items() if t == "qwen3"), reverse=True)
show("qwen3 languages, Odia rank", f'{len(q)}  rank {[l for _, l in q].index("ory_Orya") + 1}')
toks = sorted({t for t, _ in m1})
show("tokenizers", f"{len(toks)} {toks}")
show("Odia ratio per tokenizer", {t: float(m1[(t, "ory_Orya")]["ratio_to_eng"]) for t in toks if (t, "ory_Orya") in m1})

# --- M2 ops
m2 = json.load(open("m2/m2_ops.json"))["models"]["Qwen3-8B"]
s = m2["shapes"]
d, i, Lr, H, K, h, V = s["d"], s["i"], s["L"], s["H"], s["KV"], s["hd"], s["V"]
per_layer = d * H * h + 2 * d * K * h + H * h * d + 3 * d * i
lin = Lr * per_layer + d * V
show("per-layer parts (M): Q, KV, O, MLP", [round(x / 1e6, 1) for x in (d * H * h, 2 * d * K * h, H * h * d, 3 * d * i)])
show("per layer, x36, head, M_lin",
     f"{per_layer/1e6:.1f}M {Lr*per_layer/1e9:.2f}G {d*V/1e9:.2f}G {lin/1e9:.3f}G (json {m2['linear_params']/1e9:.3f}G)")
show("attn per ctx 2*l*H*h", 2 * Lr * H * h)
a8 = 2 * Lr * H * h * 8192
show("attention at 8192 and share of total", f"{a8/1e9:.2f}G  {a8/(lin+a8):.3f}")
show("json ctx rows", [p["ctx"] for p in m2["per_ctx"]])
show("k bytes/token, weights GB", f"{2*Lr*K*h*2}  {2*lin/1e9:.2f}")

# --- M4 Deng test
pl = rows("m4/per_language.csv")
for r in pl[:6]:
    show(f"  {r['language']}", f"out {float(r['out_tokens_per_resp']):.0f} pred {float(r['pred_ratio_time']):.1f} meas {float(r['meas_ratio']):.1f}")
dt = json.load(open("m4/deng_test.json"))
show("n languages", dt["n_languages"])
base = [r for r in dt["runs"] if r["p_eng"] == 220 and r["kv_budget_gb"] == 23.0][0]
show("spearman time (base), baselines", f'{base["spearman_time"]:.3f} {dt["baselines_spearman"]}')
over = sorted(((float(r["pred_ratio_time"]) / float(r["meas_ratio"]), r["language"]) for r in pl), reverse=True)[:8]
show("top over-predictions", [(n, round(x, 2)) for x, n in over])
fl = [r["max_pred_floor"] for r in dt["runs"] if r["p_eng"] == 220]
show("floor max ratio at P=220 over KV and e_bit", f"{min(fl):.2f} to {max(fl):.2f}")
show("max measured ratio", f'{base["max_meas"]:.2f}')
show("implied English power W (ideal roofline)", f'{base["implied_power_w_eng"]:.0f}')
D = rows("m4/deng2026_table5.csv")
show("deng columns", list(D[0].keys()))
jt = [float(r["total_energy_j"]) / float(r["output_tokens"]) for r in D]
tot = [float(r["total_energy_j"]) for r in D]
show("Deng J/token max/min, total max/min", f"{max(jt)/min(jt):.2f}  {max(tot)/min(tot):.1f}  (n={len(D)})")

# --- M3a
inv = {(r["corner"], float(r["vdd"])): r for r in rows("m3a/inverter_sweep.csv")}
e18, e05 = float(inv[("tt", 1.8)]["e_dyn"]) * 1e15, float(inv[("tt", 0.5)]["e_dyn"]) * 1e15
show("inverter e_dyn tt 1.8 / 0.5 (fJ), V^2 pred", f"{e18:.2f} / {e05:.2f}  pred {e18*(0.5/1.8)**2:.2f}")
vs = sorted(v for c, v in inv if c == "tt" and inv[(c, v)]["tpd"])
sl = []
for a, b in zip(vs, vs[1:]):
    ta, tb = float(inv[("tt", a)]["tpd"]), float(inv[("tt", b)]["tpd"])
    sl.append((a, b, round((ta / tb) ** (0.1 / (b - a)), 1)))
show("tt inverter delay factor per 100 mV", sl)
mac = json.load(open("m3a/mac.json"))
pts = {p["vdd"]: p for p in mac["corners"]["tt"]["points"]}
show("MAC tt 1.8 / 0.5 pJ, slowdown, saving",
     f'{pts[1.8]["e_mac"]*1e12:.3f} / {pts[0.5]["e_mac"]*1e12:.4f}  {pts[0.5]["cycle_s"]/pts[1.8]["cycle_s"]:.0f}x  '
     f'{pts[1.8]["e_mac"]/pts[0.5]["e_mac"]:.1f}x')
mep = json.load(open("m3a/mep.json"))
show("mep.json keys / tt keys", f'{list(mep.keys())} / {list(mep["corners"]["tt"].keys())}')

# --- M3b
sr = {(r["corner"], float(r["vdd"])): r for r in rows("m3b/sram_sweep.csv")}
r18, r05 = float(sr[("tt", 1.8)]["e_read"]) * 1e15, float(sr[("tt", 0.5)]["e_read"]) * 1e15
show("SRAM e_read tt 1.8 / 0.5 fJ, 156/e18", f"{r18:.2f} / {r05:.2f}  {156/r18:.1f}x")

# --- chain
ch = {r["lang"]: r for r in rows("m4/chain_per_language.csv")}
o = ch["ory_Orya"]
show("Odia token_ratio, N1_b1 ratio, share weights", f'{o["token_ratio"]} {o["N1_b1_ratio"]} {o["N1_b1_share_weights"]}')
show("Odia N20_b64 ratio [lo,hi]", f'{o["N20_b64_ratio"]} [{o["N20_b64_ratio_lo"]}, {o["N20_b64_ratio_hi"]}]')
show("Odia N200_b64 ratio [lo,hi]", f'{o["N200_b64_ratio"]} [{o["N200_b64_ratio_lo"]}, {o["N200_b64_ratio_hi"]}]')
sw = [float(r["N1_b1_share_weights"]) for r in ch.values()]
show("N1_b1 share weights range", f"{min(sw):.3f} to {max(sw):.3f}")
show("weights read 2*M_lin*8*7.5pJ", f"{2*lin*8*7.5e-12:.3f} J")
n_eng = 27.293
gm = lin * n_eng + 2 * Lr * H * h * n_eng * (n_eng + 1) / 2
show("English sentence MAC (prefill, causal attn)", f"{gm/1e9:.1f} G  linear only {lin*n_eng/1e9:.1f} G")
qe = 1.602176634e-19
for Vv in (1.8, 0.5):
    for name, mult in (("eng", 1), ("ory", 9.86)):
        E = 204.5e9 * mult * pts[Vv]["e_mac"]
        show(f"electrons {name} {Vv} V", f"E {E:.3g} J  N {E/(qe*Vv):.3g}")

# --- M5 logic
lg = {(r["corner"], float(r["vdd"])): r for r in rows("m5/logic_mc_summary.csv")}
for c in ("tt", "ss", "ff"):
    show(f"logic {c} guarded fJ", {v: round(float(lg[(c, v)]["e_mac_guarded"]) * 1e15) for v in (1.8, 1.0, 0.7, 0.6, 0.5, 0.45)})
show("logic tt nominal fJ", {v: round(float(lg[("tt", v)]["e_mac_nominal"]) * 1e15, 1) for v in (1.8, 1.0, 0.7, 0.6, 0.5, 0.45)})
show("logic tt sd/mean", {v: round(float(lg[("tt", v)]["tpd_sd_over_mean"]), 2) for v in (1.8, 1.0, 0.7, 0.5)})
show("wrong copies", {k: r["frac_wrong"] for k, r in lg.items() if float(r["frac_wrong"]) > 0})

# --- M5 SRAM
sm = {(r["corner"].replace("_mm", ""), float(r["vdd"])): r for r in rows("m5/sram_mc_summary.csv") if r["corner"].endswith("_mm")}
show("SRAM MC cells per point", sorted({r["n"] for r in sm.values()}))
for c in ("tt", "ff", "ss", "sf"):
    show(f"SRAM read fail % {c}", {v: round(float(sm[(c, v)]["read_snm_fail_frac"]) * 100, 1) for v in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)})
show("SRAM fs write fail % 1.0 / 0.6",
     f'{float(sm[("fs",1.0)]["write_fail_frac"])*100:.1f} / {float(sm[("fs",0.6)]["write_fail_frac"])*100:.1f}')
show("hold fail % ss / sf at 0.5",
     f'{float(sm[("ss",0.5)]["hold_snm_fail_frac"])*100:.1f} / {float(sm[("sf",0.5)]["hold_snm_fail_frac"])*100:.1f}')


def z(c, v, key):
    x = sm[(c, v)][key]
    return float(x) if x else 0.0


def low6(c, key):
    allv = sorted(v for (cc, v) in sm if cc == c)
    best = None
    for v in reversed(allv):
        if z(c, v, key) >= 6:
            best = v
        else:
            break
    return best


for c in ("tt", "ff", "ss", "sf", "fs"):
    show(f"SRAM mu/sd >= 6 down to V, {c} read/write", f'{low6(c, "read_snm_mu_over_sd")} / {low6(c, "write_trip_mu_over_sd")}')
show("sf read mu/sd by V", {v: round(z("sf", v, "read_snm_mu_over_sd"), 1) for (cc, v) in sorted(sm) if cc == "sf"})
N = 8 * 2**20
p = -math.log(0.99) / N
show("1 MB cells, p, z", f"{N:.3g} {p:.3g} {-NormalDist().inv_cdf(p):.2f}")
print("\n".join(out))
