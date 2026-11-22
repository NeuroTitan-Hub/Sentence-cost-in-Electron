"""Static SVG figures for the first post, drawn straight from the data files.
Run from the repo root:  python -I m6/make_figures.py   ->  m6/figures/*.svg

Palette: reference categorical slots (blue, orange, aqua, yellow), validated light-mode
(worst adjacent CVD dE 9.1). Aqua and yellow are under 3:1 on the light card, so every line
is direct-labelled and every figure has a data table in the post (relief rule).
Figures are <img> files, so they cannot follow the site's theme classes: each draws its own
light card. Colour follows the entity: typical = blue, slow = orange, fast = aqua, slow-n/fast-p = yellow.
"""
import csv
import json
import math
from pathlib import Path

OUT = Path("m6/figures")
OUT.mkdir(parents=True, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
INK, INK2, MUTED, GRID, CARD = "#151515", "#3b3b3b", "#646464", "#e4e1db", "#fbfaf7"
FONT = "IBM Plex Sans, Segoe UI, Helvetica, Arial, sans-serif"


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


class Fig:
    def __init__(self, title, subtitle, w=720, h=440, ml=64, mr=150, mt=84, mb=56):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, ml, mr, mt, mb
        self.parts = []
        self.title, self.subtitle = title, subtitle

    # axes
    def set_x(self, lo, hi, log=False):
        self.x0, self.x1, self.xlog = lo, hi, log

    def set_y(self, lo, hi, log=False):
        self.y0, self.y1, self.ylog = lo, hi, log

    def X(self, v):
        f = (math.log10(v) - math.log10(self.x0)) / (math.log10(self.x1) - math.log10(self.x0)) if self.xlog else (v - self.x0) / (self.x1 - self.x0)
        return self.ml + f * (self.w - self.ml - self.mr)

    def Y(self, v):
        f = (math.log10(v) - math.log10(self.y0)) / (math.log10(self.y1) - math.log10(self.y0)) if self.ylog else (v - self.y0) / (self.y1 - self.y0)
        return self.h - self.mb - f * (self.h - self.mt - self.mb)

    def text(self, x, y, s, size=12, color=INK2, anchor="start", weight=400):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>')

    def grid(self, xt, yt, xfmt=str, yfmt=str, xlabel="", ylabel=""):
        L, R, T, B = self.ml, self.w - self.mr, self.mt, self.h - self.mb
        for v in yt:
            y = self.Y(v)
            self.parts.append(f'<line x1="{L}" x2="{R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            self.text(L - 8, y + 4, yfmt(v), 11, MUTED, "end")
        for v in xt:
            x = self.X(v)
            self.parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{B}" y2="{B + 4}" stroke="{MUTED}" stroke-width="1"/>')
            self.text(x, B + 18, xfmt(v), 11, MUTED, "middle")
        self.parts.append(f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{MUTED}" stroke-width="1"/>')
        if xlabel:
            self.text((L + R) / 2, self.h - 14, xlabel, 12, INK2, "middle")
        if ylabel:
            self.text(L, T - 12, ylabel, 12, INK2, "start")

    def line(self, pts, color, width=2, dash=None):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.X(x):.1f},{self.Y(y):.1f}" for i, (x, y) in enumerate(pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{da}/>')

    def dot(self, x, y, color, r=4.5, ring=True):
        rs = f' stroke="{CARD}" stroke-width="2"' if ring else ""
        self.parts.append(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" fill="{color}"{rs}/>')

    def band(self, xa, xb, label, lx=None, ly=None, anchor="middle"):
        x0, x1 = self.X(xa), self.X(xb)
        self.parts.append(f'<rect x="{min(x0, x1):.1f}" y="{self.mt}" width="{abs(x1 - x0):.1f}" height="{self.h - self.mt - self.mb}" fill="#151515" fill-opacity="0.05"/>')
        self.text((x0 + x1) / 2 if lx is None else lx, self.mt + 14 if ly is None else ly, label, 11, MUTED, anchor)

    def rect(self, x, y, w, h, color):
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.2f}" height="{h:.1f}" fill="{color}"/>')

    def save(self, name, source):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'font-family="{FONT}" role="img"><title>{self.title}</title>'
                f'<rect width="{self.w}" height="{self.h}" rx="6" fill="{CARD}"/>')
        tt = [f'<text x="{self.ml - 40}" y="26" font-size="15" font-weight="600" fill="{INK}">{self.title}</text>',
              f'<text x="{self.ml - 40}" y="44" font-size="12" fill="{MUTED}">{self.subtitle}</text>',
              f'<text x="{self.w - 12}" y="{self.h - 6}" font-size="9.5" fill="{MUTED}" text-anchor="end">{source}</text>']
        (OUT / name).write_text(head + "".join(tt) + "".join(self.parts) + "</svg>\n", encoding="utf-8")
        print("wrote", OUT / name)


def fmt_x(v):
    return f"{v:g}×"


# ---------- Fig 1: token ratios, Qwen3, 204 languages
m1 = sorted((float(r["ratio_to_eng"]), r["lang"]) for r in rows("m1/summary.csv") if r["tokenizer"] == "qwen3")
f = Fig("Tokens needed for the same sentences, relative to English", "Qwen3 tokenizer, FLORES-200, 204 languages, sorted", mr=40)
f.set_x(0, len(m1))
f.set_y(0, 11)
f.grid([], [0, 2, 4, 6, 8, 10], yfmt=fmt_x, xlabel="204 languages, cheapest to most expensive", ylabel="token ratio")
bw = (f.X(1) - f.X(0))
for k, (r, l) in enumerate(m1):
    col = C["orange"] if l in ("ory_Orya", "shn_Mymr") else (INK if l == "eng_Latn" else C["blue"])
    y = f.Y(r)
    f.rect(f.X(k) + 0.3, y, max(bw - 0.6, 0.8), f.Y(0) - y, col)
names = {"eng_Latn": "English 1.00×", "ory_Orya": "Odia 9.70×", "shn_Mymr": "Shan 10.43×"}
for k, (r, l) in enumerate(m1):
    if l in names:
        if l == "eng_Latn":
            f.parts.append(f'<line x1="{f.X(k + 0.5):.1f}" x2="{f.X(k + 0.5):.1f}" y1="{f.Y(1.05):.1f}" y2="{f.Y(2.5):.1f}" stroke="{INK}" stroke-width="1"/>')
            f.text(f.X(k + 0.5) + 2, f.Y(2.5) - 6, names[l], 12, INK, "start", 600)
        else:
            f.text(f.X(k + 0.5) - 6, f.Y(r) - 6 + (14 if l == "ory_Orya" else 0), names[l], 12, INK, "end", 600)
med = m1[len(m1) // 2][0]
f.text(f.X(len(m1) / 2), f.Y(med) - 8, f"median {med:.2f}×", 11, MUTED, "middle")
f.save("fig-1-tokens.svg", "Data: m1/summary.csv")

# ---------- Fig 2: Deng test, predicted vs measured
pl = rows("m4/per_language.csv")
f = Fig("Predicted vs measured energy per token, relative to English", "117 languages; prediction from counted memory traffic, nothing fitted; rank correlation 0.94", mr=40)
f.set_x(0.8, 14, True)
f.set_y(0.8, 14, True)
ticks = [1, 2, 4, 8]
f.grid(ticks, ticks, fmt_x, fmt_x, "predicted (our model)", "measured (Deng et al., Table 5)")
f.line([(0.8, 0.8), (14, 14)], MUTED, 1, "4 4")
f.text(f.X(11), f.Y(11) - 6, "predicted = measured", 11, MUTED, "end")
lab = {"Tibetan", "Shan", "Odia", "Burmese", "Southern Pashto", "Amharic"}
for r in pl:
    f.dot(float(r["pred_ratio_time"]), float(r["meas_ratio"]), C["blue"], 4, True)
for r in pl:
    if r["language"] in lab:
        x, y = float(r["pred_ratio_time"]), float(r["meas_ratio"])
        left = r["language"] in ("Southern Pashto", "Amharic", "Odia")
        dy = {"Odia": -10, "Burmese": 12}.get(r["language"], 4)
        f.text(f.X(x) + (-12 if left else 8), f.Y(y) + dy, r["language"], 11, INK, "end" if left else "start", 500)
f.save("fig-2-deng-test.svg", "Data: m4/per_language.csv (P = 220 tokens, KV budget 23 GB)")

# ---------- Fig 3: inverter delay vs supply
inv = sorted((float(r["vdd"]), float(r["tpd"])) for r in rows("m3a/inverter_sweep.csv") if r["corner"] == "tt" and r["tpd"])
f = Fig("Inverter delay vs supply voltage", "sky130, typical corner, 27 °C, simulated transistor by transistor", mr=40)
f.set_x(0.2, 1.85)
f.set_y(1e-11, 1e-4, True)
f.grid([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
       lambda v: f"{v:g} V", lambda v: {1e-11: "10 ps", 1e-10: "100 ps", 1e-9: "1 ns", 1e-8: "10 ns", 1e-7: "100 ns", 1e-6: "1 µs", 1e-5: "10 µs", 1e-4: "100 µs"}[v],
       "supply voltage", "delay (log scale)")
f.band(0.3, 0.5, "below 0.5 V: about 9.5× slower per 100 mV", lx=f.X(0.53), ly=f.Y(3e-6), anchor="start")
f.line(inv, C["blue"])
for v, t in inv:
    f.dot(v, t, C["blue"], 3.5)
f.save("fig-3-delay.svg", "Data: m3a/inverter_sweep.csv")

# ---------- Fig 4: energy per MAC vs supply, with and without variation
lg = {(r["corner"], float(r["vdd"])): r for r in rows("m5/logic_mc_summary.csv")}
vs = [0.45, 0.5, 0.6, 0.7, 1.0, 1.8]
f = Fig("Energy per 8-bit multiply-add vs supply voltage", "sky130, 27 °C; clock set for the slowest 3 sigma of 50 simulated copies", mr=170)
f.set_x(0.4, 1.85)
f.set_y(80, 2000, True)
f.grid([0.5, 0.75, 1.0, 1.25, 1.5, 1.75], [100, 200, 500, 1000, 2000], lambda v: f"{v:g} V", lambda v: f"{v:g} fJ",
       "supply voltage", "energy per multiply-add (log scale)")
series = [("typical, one ideal copy", "tt", "e_mac_nominal", C["blue"], "5 4"),
          ("typical, with variation", "tt", "e_mac_guarded", C["blue"], None),
          ("slow corner, with variation", "ss", "e_mac_guarded", C["orange"], None),
          ("fast corner, with variation", "ff", "e_mac_guarded", C["aqua"], None)]
label_y = {}
for name, c, key, col, dash in series:
    pts = [(v, float(lg[(c, v)][key]) * 1e15) for v in vs]
    f.line(pts, col, 2, dash)
    vmin, emin = min(pts, key=lambda p: p[1])
    f.dot(vmin, emin, col, 5)
    label_y[name] = (pts[-1][1], col)
# legend at right, ordered by end value is crowded at 1.8 V, so stack fixed
ly = f.mt + 20
for name, c, key, col, dash in series:
    x = f.w - f.mr + 14
    da = f' stroke-dasharray="{dash}"' if dash else ""
    f.parts.append(f'<line x1="{x}" x2="{x + 22}" y1="{ly - 4}" y2="{ly - 4}" stroke="{col}" stroke-width="2"{da}/>')
    f.text(x + 28, ly, name, 11, INK2)
    ly += 20
f.text(f.w - f.mr + 14, ly + 10, "dot = lowest energy", 11, MUTED)
f.save("fig-4-energy.svg", "Data: m5/logic_mc_summary.csv")

# ---------- Fig 5: SRAM read failures vs supply
sm = {(r["corner"].replace("_mm", ""), float(r["vdd"])): r for r in rows("m5/sram_mc_summary.csv") if r["corner"].endswith("_mm")}
vv = sorted({v for (_, v) in sm})
f = Fig("SRAM cells that lose their bit when read", "1,000 simulated minimum-size cells per corner and supply, sky130, 27 °C", mr=150)
f.set_x(0.45, 1.85)
f.set_y(0, 100)
f.grid([0.5, 0.75, 1.0, 1.25, 1.5, 1.75], [0, 25, 50, 75, 100], lambda v: f"{v:g} V", lambda v: f"{v:g}%",
       "supply voltage", "cells failing to read")
f.band(0.5, 0.6, "← logic is cheapest here (0.5 to 0.6 V)", lx=f.X(0.62), ly=f.Y(85) + 0, anchor="start")
cs = [("typical", "tt", C["blue"]), ("slow", "ss", C["orange"]), ("fast", "ff", C["aqua"]), ("slow n, fast p", "sf", C["yellow"])]
ends = []
for name, c, col in cs:
    pts = [(v, float(sm[(c, v)]["read_snm_fail_frac"]) * 100) for v in vv]
    f.line(pts, col)
    for p in pts:
        f.dot(*p, col, 3.2)
# legend (lines overlap at 0% and 100%, so a legend, not end labels)
ly = f.mt + 20
for name, c, col in cs:
    x = f.w - f.mr + 14
    f.parts.append(f'<line x1="{x}" x2="{x + 22}" y1="{ly - 4}" y2="{ly - 4}" stroke="{col}" stroke-width="2"/>')
    f.text(x + 28, ly, name, 11, INK2)
    ly += 20
f.save("fig-5-sram.svg", "Data: m5/sram_mc_summary.csv")

# ---------- Fig 6: energy gap vs token gap, alone and busy
ch = rows("m4/chain_per_language.csv")
f = Fig("Energy gap vs token gap, reading a page and writing a short answer", "Qwen3-8B, all 204 languages; 45 nm multiply-add and GDDR6 energies", mr=150)
f.set_x(0.8, 12, True)
f.set_y(0.8, 40, True)
f.grid([1, 2, 4, 8], [1, 2, 4, 8, 16, 32], fmt_x, fmt_x, "token ratio r", "energy ratio G")
rr = [0.8 * (12 / 0.8) ** (k / 60) for k in range(61)]
f.line([(r, r) for r in rr if r <= 40], MUTED, 1, "4 4")
f.line([(r, r * r) for r in rr if 0.8 <= r * r <= 40], MUTED, 1, "4 4")
f.text(f.X(1.6) + 4, f.Y(1.6) + 16, "G = r", 11, MUTED, "start")
f.text(f.X(4.6) - 8, f.Y(21), "G = r²", 11, MUTED, "end")
for r in ch:
    t = float(r["token_ratio"])
    f.dot(t, float(r["N20_b1_ratio"]), C["blue"], 3.2, True)
    f.dot(t, float(r["N20_b64_ratio"]), C["orange"], 3.2, True)
o = [r for r in ch if r["lang"] == "ory_Orya"][0]
f.text(f.X(float(o["token_ratio"])) - 8, f.Y(float(o["N20_b64_ratio"])) - 2, "Odia 16.5×", 11, INK, "end", 600)
f.text(f.X(float(o["token_ratio"])) + 8, f.Y(float(o["N20_b1_ratio"])) + 14, "Odia 10.2×", 11, INK, "start", 600)
ly = f.mt + 20
for name, col in (("alone", C["blue"]), ("server, 64 users", C["orange"])):
    x = f.w - f.mr + 14
    f.parts.append(f'<circle cx="{x + 10}" cy="{ly - 4}" r="4.5" fill="{col}"/>')
    f.text(x + 24, ly, name, 11, INK2)
    ly += 20
f.save("fig-6-gap.svg", "Data: m4/chain_per_language.csv (20 input sentences, 5 output)")
