"""Figures for M7 experiment A, drawn from the measured files.
Usage (from blog/first-post):  python -I m7-validation/make_m7_figures.py <run dir> <analysis dir>
  fig-8-telemetry.svg      GPU power over time for one English and one Odia "copy" request, prompt phase shaded
  fig-9-measured-gap.svg   measured energy ratio against token ratio for every language, with y = x
"""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "m6")
import make_figures as MF  # noqa: E402  (also redraws figs 1-6, unchanged)
from make_figures import C, INK, MUTED, Fig  # noqa: E402

RUN, AN = Path(sys.argv[1]), Path(sys.argv[2])
MF.OUT = OUT = Path("m7-validation/figures")
OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads((RUN / "meta.json").read_text(encoding="utf-8"))
runs = list(csv.DictReader(open(RUN / "runs.csv", encoding="utf-8")))
tel = [tuple(map(float, r)) for r in list(csv.reader(open(RUN / "telemetry.csv", encoding="utf-8")))[1:]]
per = list(csv.DictReader(open(AN / "per_language.csv", encoding="utf-8")))
names = {r["code"]: r["language"] for r in csv.DictReader(open("m4/deng2026_table5.csv", encoding="utf-8"))}


def nm(code):
    return names.get(code, code)


# ---------- fig 8: telemetry of one English and one Odia copy request (repeat 2, the middle one)
def window(lang):
    reps = sorted({x["rep"] for x in runs})
    rep = "2" if "2" in reps else reps[0]
    r = next(x for x in runs if x["lang"] == lang and x["task"] == "copy" and x["rep"] == rep)
    t0, tf, t1 = float(r["t0"]), float(r["t_first"]), float(r["t_end"])
    pts = [(t - t0, w) for t, w, *_ in tel if t0 - 1 <= t <= t1 + 1]
    return r, t0, tf, t1, pts


re_, e0, ef, e1, ep = window("eng_Latn")
ro, o0, of, o1, op = window("ory_Orya")
f = Fig("GPU power while copying 20 FLORES sentences: English against Odia",
        f"{meta['gpu'].split(',')[0]}, plugged in, a 4-bit Qwen3.5-4B fine-tune via Ollama; shaded: reading the prompt", mr=170)
xmax = math.ceil((o1 - o0) / 10) * 10 + 5
f.set_x(-1, xmax)
f.set_y(0, 120)
f.grid(list(range(0, int(xmax), 10)), [0, 20, 40, 60, 80, 100, 120], lambda v: f"{v:g} s", lambda v: f"{v:g} W", "time since the request was sent", "GPU board power")
for (a, b, col) in ((0, ef - e0, C["blue"]), (0, of - o0, C["orange"])):
    x0, x1 = f.X(a), f.X(b)
    f.parts.append(f'<rect x="{x0:.1f}" y="{f.mt}" width="{x1 - x0:.1f}" height="{f.h - f.mt - f.mb}" fill="{col}" fill-opacity="0.10"/>')
f.line(ep, C["blue"], 2)
f.line(op, C["orange"], 2)
ly = f.mt + 20
for r, col, lab in ((re_, C["blue"], "English"), (ro, C["orange"], "Odia")):
    x = f.w - f.mr + 14
    f.parts.append(f'<line x1="{x}" x2="{x + 22}" y1="{ly - 4}" y2="{ly - 4}" stroke="{col}" stroke-width="2"/>')
    f.text(x + 28, ly, lab, 11, INK, weight=600)
    f.text(x + 28, ly + 15, f"{int(r['prompt_tokens']):,} in, {int(r['output_tokens']):,} out", 10.5, MUTED)
    f.text(x + 28, ly + 29, f"{float(r['energy_j']) / 1000:.2f} kJ in {float(r['wall_s']):.0f} s", 10.5, MUTED)
    ly += 52
f.save("fig-8-telemetry.svg", "Data: m7-validation/run1 (nvidia-smi, 100 ms)")

# ---------- fig 9: measured energy ratio against token ratio
f = Fig("Measured on a real GPU: energy gap against token gap", f"Copying 20 FLORES sentences, {len(per)} languages, plugged in (mean of up to 2 repeats); ratios to English", mr=40)
xs = [float(p["prompt_token_ratio"]) for p in per]
ys = [float(p["copy_energy_ratio_total"]) for p in per]
hi = max(max(xs), max(ys)) * 1.15
f.set_x(0.8, hi, True)
f.set_y(0.8, hi, True)
ticks = [t for t in (1, 2, 4, 8) if t <= hi]
f.grid(ticks, ticks, lambda v: f"{v:g}×", lambda v: f"{v:g}×", "token ratio (this model's tokenizer)", "energy ratio for the whole request")
f.line([(0.8, 0.8), (hi, hi)], MUTED, 1, "4 4")
f.text(f.X(hi / 2.6) + 6, f.Y(hi / 2.6) + 16, "energy ratio = token ratio", 11, MUTED, "start")
for p, x, y in zip(per, xs, ys):
    f.dot(x, y, C["orange"] if p["lang"] != "eng_Latn" else INK, 4.5)
lab = {"eng_Latn", "ory_Orya", "shn_Mymr", "zho_Hans", "hin_Deva", "amh_Ethi", "mya_Mymr", "tam_Taml"}
for p, x, y in zip(per, xs, ys):
    if p["lang"] in lab:
        f.text(f.X(x) + 8, f.Y(y) + 4, nm(p["lang"]), 11, INK, "start", 500)
f.save("fig-9-measured-gap.svg", "Data: m7-validation/run1/analysis")

# ---------- fig 10: energy per generated token against context
f = Fig("Energy per generated token does not grow with context here", "Same runs as above: joules per output token against mean context, plugged in", mr=40)
cx = [float(p["prompt_tokens"]) + float(p["output_tokens"]) / 2 for p in per]
jy = [float(p["j_per_output_token_total"]) for p in per]
f.set_x(0, max(cx) * 1.08)
f.set_y(0, 2.5)
f.grid([0, 2000, 4000, 6000, 8000], [0, 0.5, 1.0, 1.5, 2.0, 2.5], lambda v: f"{v:,.0f}", lambda v: f"{v:g} J",
       "mean context per request (tokens)", "GPU energy per generated token")
for p, x, y in zip(per, cx, jy):
    f.dot(x, y, C["orange"] if p["lang"] != "eng_Latn" else INK, 4.5)
for p, x, y in zip(per, cx, jy):
    if p["lang"] in ("eng_Latn", "ory_Orya", "shn_Mymr", "khm_Khmr", "kor_Hang"):
        f.text(f.X(x) + 8, f.Y(y) + 4, nm(p["lang"]), 11, INK, "start", 500)
f.save("fig-10-per-token.svg", "Data: m7-validation/run1/analysis_ac")
