"""Analyse M7 experiment A (measure_ollama.py output): real GPU energy per language.

Usage:  python -I analyze_run.py <run dir> <out dir>
Writes per_language.csv, fit.json, and two SVG figures (telemetry of one English and one Odia request; energy gap vs token gap).

Method
- Each "read" row is READS requests timed as one window, so per-request energy = energy / n_requests.
- Every request carries a fixed overhead (about 1.4 s in the smoke test, outside Ollama's prompt evaluation).
  So prompt-processing energy is fitted across languages:  E_read = a + b * prompt_tokens   (ordinary least squares)
  a = fixed energy per request, b = joules per prompt token.
- Generation energy per output token = (E_copy - a - b * prompt_tokens) / output_tokens.
- Everything is given twice: total GPU board energy, and energy above idle (idle watts measured with the model loaded).
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads((RUN / "meta.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(open(RUN / "runs.csv", encoding="utf-8")))
# The laptop's charger came out during run1: the GPU left its high-power state at T_SPLIT (epoch seconds).
# REGIME=ac keeps requests that ended before it, REGIME=battery those that started after it. Never mix the two.
import os  # noqa: E402
T_SPLIT, REGIME = float(os.environ.get("T_SPLIT", "0")), os.environ.get("REGIME", "all")
if REGIME == "ac":
    rows = [r for r in rows if float(r["t_end"]) <= T_SPLIT]
elif REGIME == "battery":
    rows = [r for r in rows if float(r["t0"]) >= T_SPLIT]
meta["regime"], meta["t_split"] = REGIME, T_SPLIT
for r in rows:
    for k in ("energy_j", "energy_above_idle_j", "wall_s", "prompt_eval_s", "eval_s", "energy_prefill_j", "energy_decode_j"):
        r[k] = float(r[k])
    r["n"] = int(r["n_requests"])
    r["prompt_tokens"] = int(r["prompt_tokens"])
    r["output_tokens"] = int(r["output_tokens"])


def ols(x, y):
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    a = my - b * mx
    ss_res = sum((c - a - b * xx) ** 2 for xx, c in zip(x, y))
    ss_tot = sum((c - my) ** 2 for c in y)
    n = len(x)
    se_b = math.sqrt(ss_res / (n - 2) / sxx)
    return a, b, 1 - ss_res / ss_tot, se_b


fits = {}
for key in ("energy_j", "energy_above_idle_j"):
    rd = [r for r in rows if r["task"] == "read"]
    a, b, r2, se = ols([r["prompt_tokens"] for r in rd], [r[key] / r["n"] for r in rd])
    fits[key] = {"fixed_j_per_request": a, "j_per_prompt_token": b, "se_j_per_prompt_token": se, "r2": r2, "n_points": len(rd)}
# time fit too: Ollama's own prompt-eval seconds per token, and the overhead outside it
rd = [r for r in rows if r["task"] == "read"]
ta, tb, tr2, _ = ols([r["prompt_tokens"] for r in rd], [r["wall_s"] / r["n"] for r in rd])
fits["wall_s_read"] = {"fixed_s_per_request": ta, "s_per_prompt_token": tb, "r2": tr2}

by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[r["lang"]][r["task"]].append(r)

per = []
for L, t in by.items():
    rd, cp = t["read"], t["copy"]
    rec = {"lang": L, "reps": len(cp),
           "prompt_tokens": st.mean(r["prompt_tokens"] for r in cp),
           "output_tokens": st.mean(r["output_tokens"] for r in cp),
           "copy_char_ratio": st.mean(float(r["copy_char_ratio"]) for r in cp),
           "tok_per_s_decode": st.mean(r["output_tokens"] / r["eval_s"] for r in cp if r["eval_s"] > 0),
           "tok_per_s_prefill": st.mean(r["prompt_tokens"] * r["n"] / r["prompt_eval_s"] for r in rd if r["prompt_eval_s"] > 0)}
    for key, tag in (("energy_j", "total"), ("energy_above_idle_j", "above_idle")):
        a, b = fits[key]["fixed_j_per_request"], fits[key]["j_per_prompt_token"]
        e_read = [r[key] / r["n"] for r in rd]
        e_copy = [r[key] for r in cp]
        gen = [(r[key] - a - b * r["prompt_tokens"]) / r["output_tokens"] for r in cp]
        rec[f"read_j_{tag}"] = st.mean(e_read)
        rec[f"copy_j_{tag}"] = st.mean(e_copy)
        rec[f"copy_j_{tag}_cv"] = st.stdev(e_copy) / st.mean(e_copy) if len(e_copy) > 1 else None
        rec[f"j_per_output_token_{tag}"] = st.mean(gen)
    per.append(rec)

eng = next(p for p in per if p["lang"] == "eng_Latn")
for p in per:
    p["prompt_token_ratio"] = p["prompt_tokens"] / eng["prompt_tokens"]
    p["output_token_ratio"] = p["output_tokens"] / eng["output_tokens"]
    for tag in ("total", "above_idle"):
        p[f"copy_energy_ratio_{tag}"] = p[f"copy_j_{tag}"] / eng[f"copy_j_{tag}"]
        p[f"j_per_output_token_ratio_{tag}"] = p[f"j_per_output_token_{tag}"] / eng[f"j_per_output_token_{tag}"]
per.sort(key=lambda p: p["prompt_token_ratio"])

with open(OUT / "per_language.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(per[0]))
    w.writeheader()
    for p in per:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})


# rank correlation: does the energy gap follow the token gap?
def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


tr = [p["prompt_token_ratio"] for p in per]
er = [p["copy_energy_ratio_total"] for p in per]
jt = [p["j_per_output_token_total"] for p in per]
ctx = [p["prompt_tokens"] + p["output_tokens"] / 2 for p in per]
summary = {
    "meta": meta, "fits": fits,
    "spearman_energy_vs_tokens": pearson(ranks(tr), ranks(er)),
    "log_slope_energy_vs_tokens": ols([math.log(x) for x in tr], [math.log(y) for y in er])[1],
    "j_per_output_token_vs_context": dict(zip(("intercept_j", "j_per_context_token", "r2", "se"), ols(ctx, jt))),
    "per_output_token_range_j": [min(jt), max(jt)],
}
(OUT / "fit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"model {meta['model']}  GPU {meta['gpu']}  Ollama {meta['ollama']}  idle {meta['idle_w_start']:.1f} -> {meta['idle_w_end']:.1f} W")
for key in ("energy_j", "energy_above_idle_j"):
    f_ = fits[key]
    print(f"read fit ({key}): {f_['fixed_j_per_request']:.1f} J fixed + {f_['j_per_prompt_token'] * 1000:.2f} mJ/prompt token (se {f_['se_j_per_prompt_token'] * 1000:.2f}), R2 {f_['r2']:.3f}")
print(f"read time fit: {fits['wall_s_read']['fixed_s_per_request']:.2f} s fixed + {fits['wall_s_read']['s_per_prompt_token'] * 1000:.3f} ms/token")
print(f"Spearman(copy energy ratio, token ratio) = {summary['spearman_energy_vs_tokens']:.3f};  log-log slope {summary['log_slope_energy_vs_tokens']:.3f}")
print(f"J per output token vs context: {summary['j_per_output_token_vs_context']}")
print("lang       in    out  tokx  copy J   Ex    J/outtok  cv   chars")
for p in per:
    print(f"{p['lang']:9s} {p['prompt_tokens']:5.0f} {p['output_tokens']:5.0f} {p['prompt_token_ratio']:5.2f} {p['copy_j_total']:7.0f} {p['copy_energy_ratio_total']:5.2f}"
          f"  {p['j_per_output_token_total']:6.3f}  {p['copy_j_total_cv'] or 0:.3f} {p['copy_char_ratio']:.2f}")
