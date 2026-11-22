"""M5 Monte Carlo on the low-voltage logic claim (M3a): with random transistor mismatch, does the full adder still
add correctly near 0.5 V, how much do its delay and energy spread, and does the multiply-add's minimum-energy supply move?

Same circuit, sizes, junction geometry, input vectors, loads and measurement as M3a (`m3a_fulladder_sweep.py`,
imported, not copied), with three changes:
  - the library corner is <corner>_mm, so every transistor gets its own random threshold and oxide offsets;
  - N copies of the energy, delay and leakage circuits sit in one deck, each copy with its own supplies;
  - each supply uses M3a's nominal clock period for that corner and supply, so a slow copy shows up as a slow copy.

Per copy: wrong outputs over M3a's 16 random input vectors, carry delay (Cin -> Cout, mean of rise and fall),
leakage (mean over the 8 input states), energy per addition (as M3a: total minus leakage over the run, / 16).

MAC composition (as M3a: 72 full adders, a 30-adder critical path, all 72 leaking for the cycle), three ways:
  nominal     the M3a result (no mismatch)
  mean        mean energy and mean delay of the copies (the typical chip, ignoring the slowest path)
  guarded     clock set so the 30-adder path meets timing with margin: 30 x mean delay + 3 x sqrt(30) x sd
              (sum of 30 independent adder delays, 3 sigma). Leakage then burns for that longer cycle.

Usage: python -I m5_logic_mc.py <ngspice_con.exe> <sky130 lib> <m3a fulladder_sweep.csv> <out dir>
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "m3a"))
import m3a_fulladder_sweep as M3A  # noqa: E402  (SUBCKTS, pwl, VECTORS, NVEC, with_junctions, TEMP_C)

CORNERS = ["tt", "ff", "ss"]
VDDS = [1.8, 1.0, 0.7, 0.6, 0.5, 0.45]
N = int(os.environ.get("MC_N", 100))
OUT_DIR = None
SEED = 303
N_FA, PATH = 72, 30
NUM = re.compile(r"^\s*(\w+)\s*=\s*([-+0-9.eE]+)", re.M)   # meas lines carry extra text after the value


def deck(lib: str, corner: str, vdd: float, per: float, n: int, seed: int) -> str:
    edge, t0 = per / 200, per / 4
    a, b, c = ([v[i] for v in M3A.VECTORS] for i in range(3))
    L = [f"* M5 logic Monte Carlo corner={corner} vdd={vdd} n={n} seed={seed}", f".option seed={seed}",
         f'.lib "{lib}" {corner}', f".option temp={M3A.TEMP_C}", M3A.SUBCKTS, f"Vdd vdd 0 {vdd}",
         f"Vra ra 0 {M3A.pwl(a, vdd, t0, per, edge)}", f"Vrb rb 0 {M3A.pwl(b, vdd, t0, per, edge)}",
         f"Vrc rc 0 {M3A.pwl(c, vdd, t0, per, edge)}",
         f"Vdc dcr 0 PULSE(0 {vdd} {t0} {edge} {edge} {per / 2 - edge} {per})",
         "Vka ka 0 0", "Vkb kb 0 0", "Vkc kc 0 0"]
    for i in range(n):
        L += [f"Ve{i} ve{i} 0 {vdd}", f"Vd{i} vd{i} 0 {vdd}", f"Vk{i} vk{i} 0 {vdd}"]
        for x in "abc":
            L += [f"XB1{x}{i} r{x} i{x}{i} vdd inv", f"XB2{x}{i} i{x}{i} e{x}{i} vdd inv"]
        L += [f"XE{i} ea{i} eb{i} ec{i} es{i} eco{i} ve{i} fa", f"XLES{i} es{i} vdd load4", f"XLEC{i} eco{i} vdd load4",
              f"XDB1{i} dcr dci{i} vdd inv", f"XDB2{i} dci{i} dcc{i} vdd inv",
              f"XD{i} vd{i} 0 dcc{i} ds{i} dco{i} vd{i} fa", f"XLDS{i} ds{i} vdd load4", f"XLDC{i} dco{i} vdd load4",
              f"XK{i} ka kb kc ks{i} kco{i} vk{i} fa", f"XLKS{i} ks{i} vdd load4", f"XLKC{i} kco{i} vdd load4"]
    tstop = t0 + M3A.NVEC * per
    vh = vdd / 2
    # keep only what the measurements read: without this a 100-copy deck needed 3.2 GB and ngspice aborted
    # .save also limits what op keeps, so the leakage currents i(Vk) must be listed too (2026-11-21: leaving them
    # out made every copy "missing k0" and cost a full rerun)
    L += [".save " + " ".join(f"i(Ve{i}) v(es{i}) v(eco{i}) v(dcc{i}) v(dco{i}) i(Vk{i})" for i in range(j, min(j + 20, n)))
          for j in range(0, n, 20)]
    C = [".control", f"tran {per / 500:.4e} {tstop:.4e}"]
    for i in range(n):
        C += [f"meas tran qe{i} integ i(Ve{i}) from={t0:.4e} to={tstop:.4e}",
              f"meas tran dr{i} trig v(dcc{i}) val={vh} td={t0 + per:.4e} rise=1 targ v(dco{i}) val={vh} td={t0 + per:.4e} rise=1",
              f"meas tran df{i} trig v(dcc{i}) val={vh} td={t0 + per:.4e} fall=1 targ v(dco{i}) val={vh} td={t0 + per:.4e} fall=1"]
        for k in range(M3A.NVEC):
            ts = t0 + (k + 1) * per - per * 0.02
            C += [f"meas tran s{k}x{i} find v(es{i}) at={ts:.4e}", f"meas tran c{k}x{i} find v(eco{i}) at={ts:.4e}"]
    C += ["destroy all"]
    for stt in range(8):
        C += [f"alter Vka dc={vdd * (stt & 1)}", f"alter Vkb dc={vdd * ((stt >> 1) & 1)}",
              f"alter Vkc dc={vdd * ((stt >> 2) & 1)}", "op"]
        C += [f"let k{stt}x{i} = i(Vk{i})" for i in range(n)]
        C += [f"print k{stt}x{i}" for i in range(n)]
    C += [".endc", ".end"]
    return M3A.with_junctions("\n".join(L + C) + "\n")


def run(ngspice: str, lib: str, corner: str, vdd: float, per: float) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        Path(d, ".spiceinit").write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")
        Path(d, "deck.cir").write_text(deck(lib, corner + "_mm", vdd, per, N, SEED))
        t0 = time.time()
        r = subprocess.run([ngspice, "-b", "deck.cir"], cwd=d, capture_output=True, text=True, timeout=86400)
        el = time.time() - t0
    if OUT_DIR is not None:   # keep the raw output, so a parsing slip never costs a rerun
        (OUT_DIR / f"stdout_{corner}_{vdd}.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    kv = {k.lower(): float(v) for k, v in NUM.findall(r.stdout)}
    rows = []
    for i in range(N):
        row = {"corner": corner, "vdd": vdd, "period": per, "copy": i}
        try:
            qe, dr, df = kv[f"qe{i}"], kv[f"dr{i}"], kv[f"df{i}"]
            kavg = sum(abs(kv[f"k{s}x{i}"]) for s in range(8)) / 8
            wrong = 0
            for k, (a, b, c) in enumerate(M3A.VECTORS):
                tot = a + b + c
                wrong += (kv[f"s{k}x{i}"] > vdd / 2) != bool(tot & 1)
                wrong += (kv[f"c{k}x{i}"] > vdd / 2) != bool(tot >> 1)
        except KeyError as e:
            row.update({"error": f"missing {e}"})
            rows.append(row)
            continue
        e_total, p_leak = -qe * vdd, kavg * vdd
        row.update({"wrong": wrong, "tpd": (dr + df) / 2, "dr": dr, "df": df, "e_total": e_total, "p_leak": p_leak,
                    "e_add": (e_total - p_leak * M3A.NVEC * per) / M3A.NVEC,
                    "leak_frac": p_leak * M3A.NVEC * per / e_total, "slow": max(abs(dr), abs(df)) > per / 20 or dr <= 0 or df <= 0,
                    "error": ""})
        rows.append(row)
    print(f"  {corner} {vdd} V: {el:.0f} s, errors {sum(1 for x in rows if x['error'])}", flush=True)
    return rows


def compose(rows: list[dict], nominal: dict) -> list[dict]:
    out = []
    for (c, v) in sorted({(r["corner"], r["vdd"]) for r in rows}, key=lambda k: (k[0], -k[1])):
        g = [r for r in rows if r["corner"] == c and r["vdd"] == v and not r["error"]]
        nom = nominal.get((c, v))
        s = {"corner": c, "vdd": v, "n_ok": len(g), "n_err": sum(1 for r in rows if r["corner"] == c and r["vdd"] == v and r["error"])}
        if not g:
            out.append(s)
            continue
        s["frac_wrong"] = sum(r["wrong"] > 0 for r in g) / len(g)
        s["frac_slow"] = sum(bool(r["slow"]) for r in g) / len(g)
        good = [r for r in g if r["wrong"] == 0 and r["dr"] > 0 and r["df"] > 0]
        if len(good) < 2:
            out.append(s)
            continue
        tp = [r["tpd"] for r in good]
        ea = [r["e_add"] for r in good]
        pl = [r["p_leak"] for r in good]
        mt, sdt = st.mean(tp), st.stdev(tp)
        me, sde = st.mean(ea), st.stdev(ea)
        mp = st.mean(pl)
        s.update({"tpd_mean": mt, "tpd_sd": sdt, "tpd_max": max(tp), "tpd_sd_over_mean": sdt / mt,
                  "e_add_mean": me, "e_add_sd": sde, "p_leak_mean": mp, "p_leak_sd": st.stdev(pl),
                  "leak_frac_mean": st.mean(r["leak_frac"] for r in good)})
        cyc_mean = PATH * mt
        cyc_guard = PATH * mt + 3 * math.sqrt(PATH) * sdt
        s["e_mac_mean"] = N_FA * me + N_FA * mp * cyc_mean
        s["e_mac_guarded"] = N_FA * me + N_FA * mp * cyc_guard
        s["cycle_guarded_s"] = cyc_guard
        if nom:
            s["tpd_nominal"] = nom["tpd"]
            s["e_mac_nominal"] = N_FA * nom["e_add"] + N_FA * nom["p_leak"] * PATH * nom["tpd"]
        out.append(s)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ngspice, lib, m3a_csv, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    global OUT_DIR
    OUT_DIR = out
    nominal = {(r["corner"], float(r["vdd"])): {k: float(r[k]) for k in ("period", "tpd", "e_add", "p_leak")}
               for r in csv.DictReader(open(m3a_csv)) if not r["error"]}
    jobs = [(c, v, nominal[(c, v)]["period"]) for c in CORNERS for v in VDDS if (c, v) in nominal]
    if os.environ.get("MC_ONLY"):   # e.g. "tt:1.8" for a quick check
        c0, v0 = os.environ["MC_ONLY"].split(":")
        jobs = [j for j in jobs if j[0] == c0 and j[1] == float(v0)]
    t0 = time.time()
    workers = int(os.environ.get("MC_WORKERS", len(jobs)))   # a 16 GB machine; cap parallel decks
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda j: run(ngspice, lib, *j), jobs))
    rows = [r for rs in res for r in rs]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "logic_mc_copies.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summ = compose(rows, nominal)
    keys = list(dict.fromkeys(k for s in summ for k in s))
    with open(out / "logic_mc_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summ)
    (out / "logic_mc_config.json").write_text(json.dumps({"corners": CORNERS, "vdds": VDDS, "n": N, "seed": SEED,
                                                          "n_fa": N_FA, "path": PATH, "wall_s": round(time.time() - t0)}, indent=2))
    for s in summ:
        if "e_mac_mean" not in s:
            print(f"{s['corner']} {s['vdd']:.2f} V: ok {s['n_ok']} err {s['n_err']} wrong {s.get('frac_wrong')}")
            continue
        print(f"{s['corner']} {s['vdd']:.2f} V: wrong {s['frac_wrong']:.2f} slow {s['frac_slow']:.2f} | tpd {s['tpd_mean']*1e9:.3g} ns"
              f" (nominal {s.get('tpd_nominal', float('nan'))*1e9:.3g}) sd/mean {s['tpd_sd_over_mean']:.2f} max {s['tpd_max']*1e9:.3g} ns"
              f" | E_add {s['e_add_mean']*1e15:.3g} fJ sd {s['e_add_sd']*1e15:.2g} | E_MAC nominal {s.get('e_mac_nominal', float('nan'))*1e15:.4g}"
              f" mean {s['e_mac_mean']*1e15:.4g} guarded {s['e_mac_guarded']*1e15:.4g} fJ")


if __name__ == "__main__":
    main()
