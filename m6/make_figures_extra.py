"""Two more figures for the first post, from the same data files.
Run from the repo root:  python -I m6/make_figures_extra.py
  fig-0-abstract.svg   the chain in one picture (visual abstract), every number computed here from the files
  fig-7-memory.svg     batch size against context length, from the post's formula b = min(256, B / (k c))
"""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "m6")
from make_figures import C, CARD, FONT, INK, INK2, MUTED, GRID, Fig, rows  # noqa: E402  (also redraws figs 1-6)

OUT = Path("m6/figures")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


# ---------- numbers for the abstract, all from files
m2 = json.load(open("m2/m2_ops.json"))["models"]["Qwen3-8B"]["shapes"]
M_LIN = m2["L"] * (m2["d"] * m2["H"] * m2["hd"] + 2 * m2["d"] * m2["KV"] * m2["hd"] + m2["H"] * m2["hd"] * m2["d"] + 3 * m2["d"] * m2["i"]) + m2["d"] * m2["V"]
ATT = 2 * m2["L"] * m2["H"] * m2["hd"]
m1 = {r["lang"]: r for r in rows("m1/summary.csv") if r["tokenizer"] == "qwen3"}
mac = {p["vdd"]: p for p in json.load(open("m3a/mac.json"))["corners"]["tt"]["points"]}
ch = {r["lang"]: r for r in rows("m4/chain_per_language.csv")}
Q = 1.602176634e-19


def per(code):
    n = round(float(m1[code]["tokens_per_sentence"]))
    M = n * M_LIN + ATT * n * (n + 1) // 2
    E = M * mac[1.8]["e_mac"]
    return {"n": n, "M": M, "E": E, "N": E / (Q * 1.8)}


en, od = per("eng_Latn"), per("ory_Orya")
gap_srv = float(ch["ory_Orya"]["N20_b64_ratio"])


def sci(x):
    e = math.floor(math.log10(x))
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{x / 10 ** e:.1f} × 10{str(e).translate(sup)}"


# ---------- fig 0: visual abstract
W, H = 760, 400
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" role="img">',
         '<title>The chain in one picture: one sentence, English against Odia</title>',
         f'<rect width="{W}" height="{H}" rx="6" fill="{CARD}"/>',
         f'<text x="24" y="32" font-size="16" font-weight="600" fill="{INK}">One sentence, English against Odia, followed down the chain</text>',
         f'<text x="24" y="52" font-size="12" fill="{MUTED}">Qwen3-8B reading one average FLORES sentence. Each step says how we know it.</text>']
steps = [
    ("Tokens", "measured", f"{float(m1['eng_Latn']['tokens_per_sentence']):.1f}", f"{float(m1['ory_Orya']['tokens_per_sentence']):.1f}", f"{float(m1['ory_Orya']['ratio_to_eng']):.2f}×"),
    ("Multiply-adds", "derived", f"{en['M'] / 1e9:.1f} billion", f"{od['M'] / 1e12:.2f} trillion", f"{od['M'] / en['M']:.2f}×"),
    ("Energy on a sky130 chip, 1.8 V", "simulated", f"{en['E']:.2f} J", f"{od['E']:.1f} J", f"{od['E'] / en['E']:.2f}×"),
    ("Charge through the supply", "derived", f"{sci(en['N'])} e", f"{sci(od['N'])} e", f"{od['N'] / en['N']:.2f}×"),
]
x0, bw, gap, top = 24, 166, 18, 78
for i, (name, how, a, b, r) in enumerate(steps):
    x = x0 + i * (bw + gap)
    parts.append(f'<rect x="{x}" y="{top}" width="{bw}" height="168" rx="6" fill="#ffffff" stroke="{GRID}"/>')
    parts.append(f'<text x="{x + 12}" y="{top + 24}" font-size="13" font-weight="600" fill="{INK}">{esc(name if len(name) < 22 else name[:name.rfind(" ", 0, 22)])}</text>')
    if len(name) >= 22:
        parts.append(f'<text x="{x + 12}" y="{top + 40}" font-size="13" font-weight="600" fill="{INK}">{esc(name[name.rfind(" ", 0, 22) + 1:])}</text>')
    parts.append(f'<text x="{x + 12}" y="{top + 58}" font-size="10.5" fill="{MUTED}" letter-spacing="0.06em">{how.upper()}</text>')
    parts.append(f'<circle cx="{x + 18}" cy="{top + 86}" r="5" fill="{C["blue"]}"/><text x="{x + 30}" y="{top + 90}" font-size="12" fill="{INK2}">English</text>')
    parts.append(f'<text x="{x + bw - 12}" y="{top + 90}" font-size="12.5" fill="{INK}" text-anchor="end" font-weight="600">{esc(a)}</text>')
    parts.append(f'<circle cx="{x + 18}" cy="{top + 114}" r="5" fill="{C["orange"]}"/><text x="{x + 30}" y="{top + 118}" font-size="12" fill="{INK2}">Odia</text>')
    parts.append(f'<text x="{x + bw - 12}" y="{top + 118}" font-size="12.5" fill="{INK}" text-anchor="end" font-weight="600">{esc(b)}</text>')
    parts.append(f'<text x="{x + 12}" y="{top + 152}" font-size="20" font-weight="600" fill="{INK}">{r}</text>')
    if i < len(steps) - 1:
        ax = x + bw + 2
        parts.append(f'<path d="M{ax},{top + 84} l{gap - 6},0 m-5,-4 l5,4 l-5,4" fill="none" stroke="{MUTED}" stroke-width="1.5"/>')
by = top + 192
parts.append(f'<rect x="24" y="{by}" width="{W - 48}" height="86" rx="6" fill="#ffffff" stroke="{GRID}"/>')
parts.append(f'<text x="40" y="{by + 26}" font-size="13" font-weight="600" fill="{INK}">On a GPU server shared by 64 users (a page of text in, a short answer out)</text>')
parts.append(f'<text x="40" y="{by + 48}" font-size="12" fill="{INK2}">Weight reads are shared, and work that depends on context grows with the square of the length.</text>')
parts.append(f'<text x="40" y="{by + 70}" font-size="12" fill="{INK2}">Odia then costs <tspan font-weight="600" fill="{INK}">{gap_srv:.1f}× English\'s energy</tspan>, against {float(m1["ory_Orya"]["ratio_to_eng"]):.1f}× the tokens. DERIVED, published energies.</text>')
parts.append(f'<text x="{W - 12}" y="{H - 8}" font-size="9.5" fill="{MUTED}" text-anchor="end">Data: m1, m2, m3a/mac.json, m4/chain_per_language.csv. "e" = electrons of charge.</text>')
parts.append("</svg>\n")
(OUT / "fig-0-abstract.svg").write_text("".join(parts), encoding="utf-8")
print("wrote", OUT / "fig-0-abstract.svg")

# ---------- fig 7: batch size against context
cfg = json.load(open("m4/chain_config.json"))["shapes"]
k, B, CAP = cfg["kv_bytes_per_tok"], 23e9, 256
pl = rows("m4/per_language.csv")
f = Fig("How many requests fit on the GPU at once", "Qwen3-8B on a 48 GB L40S with 23 GB left for KV cache: b = min(256, B / (k c))", mr=40)
f.set_x(100, 10000, True)
f.set_y(10, 400, True)
f.grid([100, 300, 1000, 3000, 10000], [10, 25, 50, 100, 256], lambda v: f"{v:,}", lambda v: f"{v:g}",
       "mean context per request, c (tokens)", "batch size b (requests); both axes log scale")
cs = [100 * (100 ** (j / 80)) for j in range(81)]
f.line([(c, min(CAP, B / (k * c))) for c in cs], C["blue"], 2)
thr = B / (CAP * k)
f.line([(thr, 10), (thr, 400)], MUTED, 1, "4 4")
f.text(f.X(thr) + 6, f.Y(330), f"c = {thr:.0f}: memory starts to limit the batch", 11, MUTED)
for r in pl:
    f.dot(float(r["ctx"]), float(r["batch"]), C["orange"], 3.2, True)
for code, name, dx, dy in (("eng_Latn", "English", 8, 16), ("ory_Orya", "Odia", 8, -8), ("shn_Mymr", "Shan", 8, -8), ("amh_Ethi", "Amharic", -10, 16)):
    r = next(x for x in pl if x["code"] == code)
    f.text(f.X(float(r["ctx"])) + dx, f.Y(float(r["batch"])) + dy, f"{name}: {float(r['batch']):.0f}", 11, INK, "end" if dx < 0 else "start", 600)
f.text(f.X(9500), f.Y(256) - 8, "vLLM's cap of 256", 11, MUTED, "end")
f.save("fig-7-memory.svg", "Data: m4/per_language.csv (117 languages in Deng et al.'s test)")
