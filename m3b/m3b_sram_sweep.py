"""M3b step 1: energy to read and write one bit of on-chip SRAM on sky130, versus supply voltage.

Circuit (same for every point):
  - one SRAM column: 128 six-transistor cells on a bit-line pair (BL, BLB)
  - cells use SkyWater's own SRAM devices from sky130_fd_pr at their only allowed sizes:
      pass gate  special_nfet_pass  W/L 0.14/0.15 um
      pull-down  special_nfet_latch W/L 0.21/0.15 um
      pull-up    special_pfet_latch W/L 0.14/0.15 um
  - source/drain junction geometry: the PDK's xschem default for nf=1 (see with_junctions);
    real SRAM layouts share contacts between neighbouring cells, so this is an estimate of bit-line
    junction capacitance, and bit-line WIRE capacitance is not included
  - precharge: two PMOS (W 1.0 um) plus an equaliser (W 0.5 um), gates driven by an ideal source
  - word line: an inverter (P 1.68 / N 0.84 um) on its own supply, driving the selected cell's two pass gates
  - write driver: an NMOS (W 1.0 um) that pulls BLB to ground, gate driven by an ideal source
  - cell 0 is accessed and starts with Q=0; the other 127 cells hold Q=1

Read (two passes):
  pass 1: the word line stays on; we measure how long BL takes to fall by DV = 100 mV (a typical
          sense-amplifier input; the sense amplifier itself is not simulated)
  pass 2: the word line is on only for that long; we integrate the charge drawn from the precharge
          supply and the word-line supply over the cycle (which includes restoring BL)
Write: BLB is pulled to 0 with the word line on; cell 0 must flip to Q=1; then the bit lines are restored.

Checks: read must not flip the cell; write must flip it; bit lines must be restored by the end of the cycle.
Leakage: DC supply current of the whole 128-cell column (cells + bit lines) in the hold state.

Usage: python m3b_sram_sweep.py <ngspice_con.exe> <sky130 lib> <out dir>
"""
from __future__ import annotations

import csv
import math
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CORNERS = ["tt", "ff", "ss"]
VDDS = [1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
TEMP_C = 27
NCELLS = 128
LUMP = False   # tested: ngspice's m= on this subcircuit did not scale the bit-line load (3.2 vs 6.8 fC), so not used
DV = 0.1


def with_junctions(text: str) -> str:
    """Add source/drain junction geometry to every sky130 MOSFET line (PDK xschem default, nf=1)."""
    def add(m):
        w = float(m.group(2))
        return (f"{m.group(0)} ad={w*0.29:.4g} as={w*0.29:.4g} pd={2*(w+0.29):.4g} ps={2*(w+0.29):.4g}"
                f" nrd={0.29/w:.4g} nrs={0.29/w:.4g}")
    return re.sub(r"(sky130_fd_pr__\w+fet\w*) w=([0-9.]+) l=[0-9.]+", add, text)


CELL = """
.subckt cell bl blb wl vc
XPU1 q qb vc vc sky130_fd_pr__special_pfet_latch w=0.14 l=0.15
XPD1 q qb 0 0 sky130_fd_pr__special_nfet_latch w=0.21 l=0.15
XPU2 qb q vc vc sky130_fd_pr__special_pfet_latch w=0.14 l=0.15
XPD2 qb q 0 0 sky130_fd_pr__special_nfet_latch w=0.21 l=0.15
XPG1 bl wl q 0 sky130_fd_pr__special_nfet_pass w=0.14 l=0.15
XPG2 blb wl qb 0 sky130_fd_pr__special_nfet_pass w=0.14 l=0.15
.ends
"""


def deck(lib: str, corner: str, vdd: float, per: float, mode: str, t_wl: float) -> str:
    """mode: 'read' or 'write'. Word-line input falls at 0.25 per and rises t_wl later."""
    edge = per / 1000
    t_pre_off, t_wl_on, t_pre_on = 0.20 * per, 0.25 * per, 0.80 * per
    t_wl_off = t_wl_on + t_wl
    L = [f"* M3b SRAM column {mode} corner={corner} vdd={vdd}", f'.lib "{lib}" {corner}', f".option temp={TEMP_C}", CELL,
         f"Vc vc 0 {vdd}", f"Vpre vpre 0 {vdd}", f"Vwl vwl 0 {vdd}",
         # precharge gate (active low): on, off at t_pre_off, on again at t_pre_on
         f"Vpg pg 0 PWL(0 0 {t_pre_off:.4e} 0 {t_pre_off+edge:.4e} {vdd} {t_pre_on:.4e} {vdd} {t_pre_on+edge:.4e} 0)",
         "XPRE1 bl pg vpre vpre sky130_fd_pr__pfet_01v8 w=1.0 l=0.15",
         "XPRE2 blb pg vpre vpre sky130_fd_pr__pfet_01v8 w=1.0 l=0.15",
         "XEQ bl pg blb vpre sky130_fd_pr__pfet_01v8 w=0.5 l=0.15",
         # word-line driver: inverter, input high = word line off
         f"Vwi wi 0 PWL(0 {vdd} {t_wl_on:.4e} {vdd} {t_wl_on+edge:.4e} 0 {t_wl_off:.4e} 0 {t_wl_off+edge:.4e} {vdd})",
         "XWP wl wi vwl vwl sky130_fd_pr__pfet_01v8 w=1.68 l=0.15",
         "XWN wl wi 0 0 sky130_fd_pr__nfet_01v8 w=0.84 l=0.15",
         "X0 bl blb wl vc cell"]
    if LUMP:   # the idle cells are identical (same data, word line off): one instance with a multiplier
        L += [f"XI bl blb 0 vc cell m={NCELLS - 1}"]
    else:
        L += [f"X{k} bl blb 0 vc cell" for k in range(1, NCELLS)]
    if mode == "write":
        L += [f"Vwe we 0 PWL(0 0 {t_pre_off+2*edge:.4e} 0 {t_pre_off+3*edge:.4e} {vdd} {t_wl_off:.4e} {vdd} {t_wl_off+edge:.4e} 0)",
              "XWD blb we 0 0 sky130_fd_pr__nfet_01v8 w=1.0 l=0.15"]
    idle = ["xi"] if LUMP else [f"x{k}" for k in range(1, NCELLS)]
    ics = [f"v(x0.q)=0", f"v(x0.qb)={vdd}"] + [f"v({x}.q)={vdd} v({x}.qb)=0" for x in idle]
    L += [".ic " + " ".join(ics[:2])] + [".ic " + s for s in ics[2:]]
    # same states as DC guesses, so the hold-state operating point lands in the stored data, not the metastable point
    L += [".nodeset " + " ".join(ics[:2])] + [".nodeset " + s for s in ics[2:]]
    vh = vdd / 2
    t_end = 1.2 * per   # [per, 1.2 per] is a quiet window: precharge on, word line off, nothing switching
    ctl = [".control", f"tran {per / 4000:.4e} {t_end:.4e}",
           f"meas tran qpre integ i(Vpre) from={t_pre_off:.4e} to={per:.4e}",
           f"meas tran qwl integ i(Vwl) from={t_pre_off:.4e} to={per:.4e}",
           f"meas tran qc integ i(Vc) from={t_pre_off:.4e} to={per:.4e}",
           f"meas tran twl when v(wl)={vh} rise=1",
           f"meas tran blmin min v(bl) from={t_pre_off:.4e} to={t_pre_on:.4e}",
           f"meas tran blbmin min v(blb) from={t_pre_off:.4e} to={t_pre_on:.4e}",
           f"meas tran qend find v(x0.q) at={per * 0.99:.4e}",
           f"meas tran blend find v(bl) at={per * 0.99:.4e}",
           f"meas tran blbend find v(blb) at={per * 0.99:.4e}"]
    if mode == "read":
        ctl += [f"meas tran tdv when v(bl)={vdd - DV} fall=1"]
    else:
        ctl += ["let tdv = 0"]
    ctl += ['set out = "R qpre=$&qpre qwl=$&qwl qc=$&qc twl=$&twl blmin=$&blmin blbmin=$&blbmin'
            ' qend=$&qend blend=$&blend blbend=$&blbend tdv=$&tdv"']
    # hold-state leakage: DC, precharge on, word line off; .nodeset holds every cell in its stored state
    ctl += ["alter Vpg dc=0", f"alter Vwi dc={vdd}", "op",
            'set out = "$out ic=$&i(Vc) ipre=$&i(Vpre) iwl=$&i(Vwl) q0=$&v(x0.q) q1=$&v(x1.q)"',
            "echo $out", "quit", ".endc", ".end"]
    return "\n".join(L + ctl) + "\n"


def run_deck(ngspice: str, text: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / ".spiceinit").write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")
        (Path(d) / "deck.cir").write_text(with_junctions(text))
        r = subprocess.run([ngspice, "-b", "deck.cir"], cwd=d, capture_output=True, text=True, timeout=7200)
    low = r.stdout.lower()
    i0 = low.find("r qpre=")
    i1 = low.find("ngspice-47 done", i0)
    if i0 < 0:
        return {"error": "no result: " + (r.stdout + r.stderr)[-300:]}
    kv = dict(re.findall(r"(\w+)=(\S+)", low[i0:i1 if i1 > 0 else None]))
    out = {}
    for k, v in kv.items():
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = None
    return out


def run_point(ngspice: str, lib: str, corner: str, vdd: float) -> dict:
    """If the bit lines were not restored in time, that is our cycle being too short, not the circuit: retry 4x longer."""
    per = 8e-9 * math.exp(max(0.0, 0.9 - vdd) / 0.07)
    for _ in range(3):
        r = _run_point(ngspice, lib, corner, vdd, per)
        if "not restored" not in r.get("error", ""):
            return r
        per = r["period"] * 4
    return r


def _run_point(ngspice: str, lib: str, corner: str, vdd: float, per: float) -> dict:
    row = {"corner": corner, "vdd": vdd}
    for attempt in range(5):
        p1 = run_deck(ngspice, deck(lib, corner, vdd, per, "read", 0.5 * per))
        tdv, twl = p1.get("tdv"), p1.get("twl")
        if not p1.get("error") and tdv and twl and tdv - twl < 0.2 * per:
            break
        per *= 4
    else:
        return {**row, "error": f"pass 1: bit line never fell by {DV} V: {str(p1)[:200]}"}
    t_access = tdv - twl
    t_wl_in = tdv - 0.25 * per  # from word-line input edge to DV reached
    rd = run_deck(ngspice, deck(lib, corner, vdd, per, "read", t_wl_in))
    wr = run_deck(ngspice, deck(lib, corner, vdd, per, "write", 0.5 * per))
    err = []
    for name, r in (("read", rd), ("write", wr)):
        if r.get("error") or any(r.get(k) is None for k in ("qpre", "qwl", "qend", "blend", "blbend")):
            err.append(f"{name}: missing: {str(r)[:150]}")
    if not err:
        if rd["qend"] > vdd / 2:
            err.append("read flipped the cell")
        if wr["qend"] < vdd / 2:
            err.append("write failed to flip the cell")
        for name, r in (("read", rd), ("write", wr)):
            if min(r["blend"], r["blbend"]) < 0.98 * vdd:
                err.append(f"{name}: bit lines not restored")
    if err:
        return {**row, "period": per, "error": "; ".join(err)}
    win = 0.8 * per
    if not (rd.get("q0", 1) < vdd / 2 < (rd.get("q1") or 0)):
        return {**row, "period": per, "error": f"hold op not in stored state: q0={rd.get('q0')} q1={rd.get('q1')}"}
    i_hold = abs(rd["ic"]) + abs(rd["ipre"]) + abs(rd["iwl"])
    p_leak = i_hold * vdd
    e_rd_total = -(rd["qpre"] + rd["qwl"] + rd["qc"]) * vdd
    e_wr_total = -(wr["qpre"] + wr["qwl"] + wr["qc"]) * vdd
    return {**row, "period": per, "attempts": attempt + 1, "t_access": t_access,
            "bl_swing_read": vdd - rd["blmin"], "blb_swing_write": vdd - wr["blbmin"],
            "e_read_total": e_rd_total, "e_write_total": e_wr_total, "p_leak_column": p_leak,
            "e_read": e_rd_total - p_leak * win, "e_write": e_wr_total - p_leak * win,
            "leak_frac_read": p_leak * win / e_rd_total, "error": ""}


def main() -> None:
    ngspice, lib, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    jobs = [(c, v) for c in CORNERS for v in VDDS]
    with ThreadPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(lambda cv: run_point(ngspice, lib, *cv), jobs))
    keys = ["corner", "vdd", "period", "attempts", "t_access", "bl_swing_read", "blb_swing_write", "e_read_total",
            "e_write_total", "p_leak_column", "e_read", "e_write", "leak_frac_read", "error"]
    with open(out / "sram_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in res:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"{len(res)} points in {time.time() - t:.0f} s; errors: {sum(1 for r in res if r.get('error'))}")


if __name__ == "__main__":
    main()
