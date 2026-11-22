"""M5 Monte Carlo: does the SRAM cell still hold, read and write when every transistor is a little different?

M3b simulated one ideal cell per corner. Real chips have random mismatch between neighbouring transistors, and that is
what sets SRAM's low-voltage limit. SkyWater's library models it: the *_mm corners (tt_mm, ff_mm, ...) set
MC_MM_SWITCH=1, which adds AGAUSS(0,1) x slope/sqrt(W L) to each transistor's vth0, tox, etc. It is evaluated per
instance, so every copy of a cell in one deck gets its own random draw (checked: 5 copies differ, and a new seed gives
new values).

So one deck holds N independent cells, and every cell sees the same ideal sources. Per cell and supply we measure:
  hold SNM   static noise margin with the word line off (butterfly curves, largest square, Seevinck 1987 method)
  read SNM   the same with the word line on and both bit lines at VDD: the cell must survive being read
  write trip (static) word line on, BLB at VDD, BL lowered from VDD: the BL voltage at which the old state stops
             existing. Computed from open-loop curves, so no simulator memory is involved: at each BL value the
             two half-cell curves f1 (q from qb) and f2 (qb from q) give G(q) = f1(f2(q)) - q, whose roots are the
             cell's DC states. The old state (q high) exists while G has a positive hump above its lowest root;
             the trip is where that hump height crosses zero (linear interpolation between BL steps).
             No trip by BL = 0 V means the cell cannot be written at all.
  write time (dynamic) word line on, BL stepped to 0: time until q falls through VDD/2. Measured up to a budget of
             50 x T0(VDD); "not within budget" is reported as such, not as a hard failure.
The loop is opened or closed with behavioural sources: in1 = k*V(qb) + (1-k)*V(x),
in2 = k*V(q) + (1-k)*(V(x) + V(fy)). k=0 gives the two half-cell curves for the butterfly; k=1 is the real
cross-coupled cell. Gates draw no DC current, so this changes nothing else.
Forcing the starting state: with k=0, x=0 and fy=VDD the two inverters are driven to q=1, qb=0; then k goes to 1
and the latch holds the state on its own.
  dynamic: the forcing sequence in time, scaled by T0(VDD), because near 0.5 V these high-threshold devices take
  around 100 us to move a node (sub-threshold currents into femtofarads).
Tried first and dropped: (1) a DC sweep with .nodeset, which did not hold the starting state even for identical
nominal cells; (2) a slow BL ramp, which never settled: at 0.5 V the trip kept moving from 5 us to 1.3 ms ramps;
(3) a closed-loop DC sweep from a forced state: when one cell flipped, the solver restarted and other cells lost
their state (different random cells all "flipped" at the same BL step).

Cell: SkyWater SRAM devices, same sizes as M3b (pass 0.14/0.15, latch n 0.21/0.15, latch p 0.14/0.15 um),
same junction geometry. 27 C.

Usage: python -I m5_sram_mc.py <ngspice_con.exe> <sky130 lib> <out dir>
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

import numpy as np

CORNERS = ["tt", "ff", "ss", "sf", "fs"]   # each run as <corner>_mm (mismatch) and <corner> (nominal check)
VDDS = [1.8, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
N_PER_DECK = 500
SEEDS = [101, 202]                           # 2 decks per corner -> 1,000 cells
NPTS = 200
WPTS, WBL = 100, 25   # static write: x points per curve, BL steps from VDD to 0
T0 = {1.8: 2e-9, 1.4: 2e-9, 1.2: 3e-9, 1.0: 10e-9, 0.9: 30e-9, 0.8: 100e-9, 0.7: 500e-9, 0.6: 3e-6, 0.5: 20e-6}
TMAX_DIV = 50    # max transient step = T0/50, so the write time is resolved to 2% of T0
BUDGET_T0 = 50   # dynamic write budget after the BL step, in units of T0
TEMP_C = 27


def with_junctions(text: str) -> str:
    def add(m):
        w = float(m.group(2))
        return (f"{m.group(0)} ad={w*0.29:.4g} as={w*0.29:.4g} pd={2*(w+0.29):.4g} ps={2*(w+0.29):.4g}"
                f" nrd={0.29/w:.4g} nrs={0.29/w:.4g}")
    return re.sub(r"(sky130_fd_pr__\w+fet\w*) w=([0-9.]+) l=[0-9.]+", add, text)


CELL = """
.subckt cellmc bl blb wl vc x k fy
B1 in1 0 V = V(k)*V(qb) + (1-V(k))*V(x)
B2 in2 0 V = V(k)*V(q) + (1-V(k))*(V(x) + V(fy))
XPU1 q in1 vc vc sky130_fd_pr__special_pfet_latch w=0.14 l=0.15
XPD1 q in1 0 0 sky130_fd_pr__special_nfet_latch w=0.21 l=0.15
XPU2 qb in2 vc vc sky130_fd_pr__special_pfet_latch w=0.14 l=0.15
XPD2 qb in2 0 0 sky130_fd_pr__special_nfet_latch w=0.21 l=0.15
XPG1 bl wl q 0 sky130_fd_pr__special_nfet_pass w=0.14 l=0.15
XPG2 blb wl qb 0 sky130_fd_pr__special_nfet_pass w=0.14 l=0.15
.ends
"""


def deck(lib: str, corner: str, seed: int, n: int) -> str:
    L = [f"* M5 SRAM Monte Carlo corner={corner} seed={seed} n={n}", f".option seed={seed}", f".option temp={TEMP_C}",
         f'.lib "{lib}" {corner}', with_junctions(CELL),
         # DC sources altered per supply/test; time shapes scale with T0 (source vt0). Every term that uses `time`
         # is multiplied by V(tr), which is 1 only during the transient: in a DC sweep ngspice sets `time` to the
         # sweep value (found the hard way), so without this gate the DC sweeps were corrupted. In DC: k=0,
         # fy=0, bl=VDD*(1-s).
         "vvc vc 0 1.8", "vwl wl 0 0", "vx x 0 0", "vs s 0 0", "vt0 t0 0 20n", "vtr tr 0 0",
         "Bk k 0 V = V(tr)*max(0, min(1, (time - 2*V(t0))/(2*V(t0))))",
         "Bfy fy 0 V = V(vc)*V(tr)*min(1, time/(0.1*V(t0)))",
         "Bbl bl 0 V = V(vc)*(1 - V(s))*(1 - V(tr)*max(0, min(1, (time - 5*V(t0))/(0.01*V(t0)))))",
         "Bblb blb 0 V = V(vc)"]
    L += [f"xc{i} bl blb wl vc x k fy cellmc" for i in range(n)]
    L += [".save v(x) v(bl) " + " ".join(f"v(xc{i}.q) v(xc{i}.qb)" for i in range(j, min(j + 50, n)))
          for j in range(0, n, 50)]
    L += [".control", "set filetype=binary"]
    for vi, v in enumerate(VDDS):
        stp, t0 = v / NPTS, T0[v]
        tstop = (5 + BUDGET_T0) * t0
        L += [f"alter vvc dc={v}", f"alter vt0 dc={t0:.6g}", "alter vs dc=0",
              "alter vwl dc=0", f"dc vx 0 {v} {stp:.6g}", f"write hold_{vi}.raw",
              f"alter vwl dc={v}", f"dc vx 0 {v} {stp:.6g}", f"write read_{vi}.raw",
              f"dc vx 0 {v} {v / WPTS:.6g} vs 0 1 {1 / WBL:.6g}", f"write wstat_{vi}.raw", "alter vs dc=0",
              "alter vtr dc=1", f"tran {t0 / TMAX_DIV:.6g} {tstop:.6g} 0 {t0 / TMAX_DIV:.6g}", "alter vtr dc=0"]
        for i in range(n):
            L += [f"meas tran pre{vi}_{i} find v(xc{i}.q) at={4.9 * t0:.6g}",
                  f"meas tran preb{vi}_{i} find v(xc{i}.qb) at={4.9 * t0:.6g}",
                  f"meas tran tw{vi}_{i} when v(xc{i}.q)={v / 2:.6g} fall=1"]   # td= makes ngspice 47 fail here; checked below
    L += [".endc", ".end"]
    return "\n".join(L) + "\n"


def read_raw(path: Path) -> dict[str, np.ndarray]:
    """ngspice binary rawfile, real data."""
    b = path.read_bytes()
    k = b.index(b"Binary:\n")
    head = b[:k].decode("latin-1").splitlines()
    nv = int(next(l for l in head if l.startswith("No. Variables")).split(":")[1])
    npnt = int(next(l for l in head if l.startswith("No. Points")).split(":")[1])
    iv = head.index("Variables:")
    names = [head[iv + 1 + j].split()[1].lower() for j in range(nv)]
    a = np.frombuffer(b[k + 8:k + 8 + 8 * nv * npnt], dtype="<f8").reshape(npnt, nv)
    return {nm: a[:, j] for j, nm in enumerate(names)}


def snm(x: np.ndarray, q: np.ndarray, qb: np.ndarray) -> float:
    """Static noise margin from butterfly curves (Seevinck 1987). Axes: horizontal = qb-side, vertical = q-side.
    Curve A: (x, q(x)), the inverter driving q. Curve B: (qb(x), x), the other inverter, mirrored.
    A square's diagonal lies on a line h - v = d; both curves cross each such line once, so its side is
    side(d) = hA(d) - hB(d). In a bistable cell side > 0 in the upper-left lobe (d below the metastable point) and
    side < 0 in the lower-right lobe. In a cell with one state the signs are the other way round, so taking the
    largest |side| on each side of a crossing is wrong (a first version did that and called dead cells healthy).
    SNM = max over split points k of min(max(side[:k]), max(-side[k:])): the two lobes for a bistable cell,
    negative for a cell that has only one state."""
    f1 = np.minimum.accumulate(q)            # enforce non-increasing (removes numerical wiggles at the rails)
    f2 = np.minimum.accumulate(qb)
    dA = x - f1                              # increasing in x
    dB = f2 - x                              # decreasing in x
    d = np.linspace(max(dA.min(), dB.min()), min(dA.max(), dB.max()), 4000)
    hA = np.interp(d, dA, x)
    hB = np.interp(d, dB[::-1], f2[::-1])
    side = hA - hB
    pre = np.maximum.accumulate(side)                # max(side[:k+1])
    suf = np.maximum.accumulate((-side)[::-1])[::-1]  # max(-side[k:])
    return float(np.max(np.minimum(pre[:-1], suf[1:])))


def old_state_hump(x: np.ndarray, q: np.ndarray, qb: np.ndarray) -> float:
    """Height of G(q) = f1(f2(q)) - q above its lowest root. > 0: the old state (q high) still exists."""
    f1 = np.minimum.accumulate(q)            # q as a function of in1 (= qb when the loop is closed)
    f2 = qb                                  # qb as a function of in2 (= q)
    g = np.interp(f2, x, f1) - x             # evaluated at q = x
    neg = np.nonzero(g < 0)[0]
    if len(neg) == 0:
        return float(g.min())                # never below zero: only a high state exists (stuck)
    return float(g[neg[0]:].max())


def write_trip(W: dict, cell: int, vdd: float) -> tuple[float, str]:
    q, qb = W[f"v(xc{cell}.q)"].reshape(WBL + 1, -1), W[f"v(xc{cell}.qb)"].reshape(WBL + 1, -1)
    x, bl = W["v(x)"].reshape(WBL + 1, -1), W["v(bl)"].reshape(WBL + 1, -1)[:, 0]
    h = np.array([old_state_hump(x[j], q[j], qb[j]) for j in range(WBL + 1)])
    if h[0] <= 0:
        return float("nan"), "no_old_state"  # the q-high state does not exist even with BL at VDD (read upset)
    j = np.nonzero(h <= 0)[0]
    if len(j) == 0:
        return float("nan"), "no_flip"
    j = j[0]
    t = h[j - 1] / (h[j - 1] - h[j])
    return float(bl[j - 1] + t * (bl[j] - bl[j - 1])), "ok"


MEAS = re.compile(r"^\s*(pre|preb|tw)(\d+)_(\d+)\s*=\s*([-+0-9.eE]+)", re.M)


def run(ngspice: str, lib: str, corner: str, seed: int, n: int) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        Path(d, "deck.cir").write_text(deck(lib, corner, seed, n))
        t0 = time.time()
        r = subprocess.run([ngspice, "-b", "deck.cir"], cwd=d, capture_output=True, text=True, timeout=36000)
        el = time.time() - t0
        meas = {(m[0], int(m[1]), int(m[2])): float(m[3]) for m in MEAS.findall(r.stdout)}
        rows = []
        for vi, v in enumerate(VDDS):
            files = {m: Path(d, f"{m}_{vi}.raw") for m in ("hold", "read", "wstat")}
            if not all(f.exists() for f in files.values()):
                raise RuntimeError(f"{corner} seed {seed} vdd {v}: missing output: {r.stdout[-2000:]} {r.stderr[-2000:]}")
            H, R, W = (read_raw(files[m]) for m in ("hold", "read", "wstat"))
            for i in range(n):
                q, qb = f"v(xc{i}.q)", f"v(xc{i}.qb)"
                wt, ws = write_trip(W, i, v)
                pre, preb = meas.get(("pre", vi, i)), meas.get(("preb", vi, i))
                tw = meas.get(("tw", vi, i))
                if pre is None or preb is None:
                    dyn = "meas_missing"
                elif pre <= preb:
                    dyn = "started_flipped"
                elif tw is not None and tw < 5 * T0[v] - T0[v] / TMAX_DIV:
                    dyn = "dipped_before_step"
                else:
                    dyn = "ok" if tw is not None else "not_within_budget"
                rows.append({"corner": corner, "seed": seed, "cell": i, "vdd": v,
                             "hold_snm": round(snm(H["v(x)"], H[q], H[qb]), 5),
                             "read_snm": round(snm(R["v(x)"], R[q], R[qb]), 5),
                             "write_trip": None if math.isnan(wt) else round(wt, 5), "write_status": ws,
                             "write_time_s": (tw - 5 * T0[v]) if (tw is not None and dyn == "ok") else None,
                             "write_dyn_status": dyn, "budget_s": BUDGET_T0 * T0[v]})
    print(f"  {corner} seed {seed} n {n}: {el:.0f} s", flush=True)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["corner"], r["vdd"]) for r in rows}, key=lambda k: (k[0], -k[1]))
    for c, v in keys:
        g = [r for r in rows if r["corner"] == c and r["vdd"] == v]
        s = {"corner": c, "vdd": v, "n": len(g)}
        for m in ("hold_snm", "read_snm"):
            xs = [r[m] for r in g]
            mu, sd = st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)
            s[f"{m}_mean"], s[f"{m}_sd"], s[f"{m}_min"] = round(mu, 4), round(sd, 4), round(min(xs), 4)
            s[f"{m}_fail_frac"] = round(sum(x <= 0 for x in xs) / len(xs), 4)
            s[f"{m}_mu_over_sd"] = round(mu / sd, 2) if sd > 0 else None
        ok = [r["write_trip"] for r in g if r["write_status"] == "ok"]
        s["write_fail_frac"] = round(sum(r["write_status"] == "no_flip" for r in g) / len(g), 4)
        s["write_no_old_state_frac"] = round(sum(r["write_status"] == "no_old_state" for r in g) / len(g), 4)
        if ok:
            mu = st.mean(ok)
            sd = st.stdev(ok) if len(ok) > 1 else 0.0
            s["write_trip_mean"], s["write_trip_sd"], s["write_trip_min"] = round(mu, 4), round(sd, 4), round(min(ok), 4)
            s["write_trip_mu_over_sd"] = round(mu / sd, 2) if sd > 0 else None
        tw = sorted(r["write_time_s"] for r in g if r["write_dyn_status"] == "ok")
        for k in ("ok", "not_within_budget", "started_flipped", "dipped_before_step", "meas_missing"):
            s[f"wdyn_{k}_frac"] = round(sum(r["write_dyn_status"] == k for r in g) / len(g), 4)
        s["wdyn_budget_s"] = g[0]["budget_s"]
        if tw:
            s["wdyn_median_s"] = tw[len(tw) // 2]
            s["wdyn_p99_s"] = tw[min(len(tw) - 1, int(0.99 * len(tw)))]
            s["wdyn_max_s"] = tw[-1]
        out.append(s)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ngspice, lib, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c + "_mm", s, N_PER_DECK) for c in CORNERS for s in SEEDS] + [(c, 1, 2) for c in CORNERS]
    t0 = time.time()
    workers = int(os.environ.get("MC_WORKERS", len(jobs)))   # a 16 GB machine; cap parallel decks
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda j: run(ngspice, lib, *j), jobs))
    rows = [r for rs in res for r in rs]
    with open(out / "sram_mc_cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = summarise(rows)
    with open(out / "sram_mc_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for s in summ for k in s}, key=lambda k: list(summ[0]).index(k) if k in summ[0] else 99))
        w.writeheader()
        w.writerows(summ)
    (out / "sram_mc_config.json").write_text(json.dumps({"corners": CORNERS, "vdds": VDDS, "n_per_deck": N_PER_DECK,
                                                         "seeds": SEEDS, "npts": NPTS, "temp_c": TEMP_C,
                                                         "wall_s": round(time.time() - t0)}, indent=2))
    for s in summ:
        print(f"{s['corner']:6s} {s['vdd']:4.1f} V  n={s['n']:4d} | hold {s['hold_snm_mean']*1e3:6.1f}±{s['hold_snm_sd']*1e3:5.1f} mV"
              f" | read {s['read_snm_mean']*1e3:6.1f}±{s['read_snm_sd']*1e3:5.1f} mV fail {s['read_snm_fail_frac']:.3f}"
              f" mu/sd {s['read_snm_mu_over_sd']}"
              f" | write trip {s.get('write_trip_mean', float('nan'))*1e3:6.1f}±{s.get('write_trip_sd', float('nan'))*1e3:5.1f} mV"
              f" fail {s['write_fail_frac']:.3f} no_old {s['write_no_old_state_frac']:.3f}"
              f" | write time median {s.get('wdyn_median_s', float('nan')):.3g} s max {s.get('wdyn_max_s', float('nan')):.3g} s"
              f" not in {s['wdyn_budget_s']:.2g} s: {s['wdyn_not_within_budget_frac']:.3f}"
              f" startflip {s['wdyn_started_flipped_frac']:.3f} missing {s['wdyn_meas_missing_frac']:.3f}")


if __name__ == "__main__":
    main()
