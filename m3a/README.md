# M3a step 1: sky130 inverter versus supply voltage, 2026-11-19

Status: done for the inverter; Reproducibility checks applied (below). Next in M3a: full adder and 8-bit multiply-add.

> **Correction, 2026-11-19 evening: junction capacitance.** The sky130 device subcircuits default to zero source/drain junction area (`ad=as=pd=ps=0`). Our first runs used that default and so left out junction capacitance. Both sweeps now add the PDK's own xschem default geometry (nf=1: `ad=as=W*0.29`, `pd=ps=2*(W+0.29)`). We reran everything; the first runs are kept as `*_nojunc.*` for comparison. **The tables below this box are from the first runs.** The corrected headline numbers are:
>
> | | First run (no junctions) | Corrected |
> |---|---|---|
> | Full adder, energy per addition, tt 1.8 V | 14.9 fJ | **17.3 fJ** |
> | Full adder, energy per addition, tt 0.5 V | 0.82 fJ | **1.05 fJ** |
> | Full adder, carry delay, tt / ff / ss at 1.8 V | 136 / 99 / 204 ps | **154 / 111 / 235 ps** |
> | 8-bit MAC, tt 1.8 V | 1.07 pJ | **1.25 pJ** |
> | 8-bit MAC at the lowest resolvable supply, tt (0.5 V) | 80 fJ | **101 fJ** (12.3x less, about 4,900x slower) |
> | 8-bit MAC, ff (best at 0.6 V) / ss (0.5 V) | 145 / 85 fJ | **161 / 109 fJ** |
> | Inverter MEP model, tt, alpha 0.1 | 0.50 V, 14.8x less | **0.50 V, 14.3x less** |
>
> The conclusions do not change: the minimum-energy point is at or below about 0.5 V, at the edge of what we can resolve, and the same points fail (ff at 0.3-0.35 V for the adder; the inverter's lowest supplies). Junctions add 15-30% to energy and 13-27% to delay. Run times were 2,847 s (inverter) and 2,721 s (adder), slower because three sweeps shared the CPU.

## Setup
- ngspice 47, sky130 PDK (volare c6d73a35…, `sky130_fd_pr`), 27 C, corners tt / ff / ss.
- Fan-out-of-4 inverter chain, DUT is stage 3 with its own supply; NMOS 0.42/0.15 um, PMOS 0.84/0.15 um.
- 16 supply points 1.8 V to 0.2 V. Input period grows with delay; a point is retried with a 4x longer period until both edges settle within 5% of the period.
- Code: `m3a_inverter_sweep.py` (simulation), `m3a_analyze.py` (minimum-energy point). Data: `inverter_sweep.csv`, `mep.json`. Run time 542 s on the laptop.

## Results: the inverter
| | tt | ff | ss |
|---|---|---|---|
| FO4 delay at 1.8 V | 66 ps | 48 ps | 97 ps |
| FO4 delay at 0.5 V | 264 ns | 30 ns | 3.1 us |
| Switching energy per transition at 1.8 V | 11.6 fJ | 11.6 fJ | 11.6 fJ |
| Switching energy at 0.5 V | 0.63 fJ | 0.63 fJ | 0.63 fJ |
| Leakage power at 1.8 V | 0.59 nW | 16.5 nW | 0.004 nW |

Switching energy falls close to V^2 (11.6 fJ x (0.5/1.8)^2 = 0.89 fJ predicted vs 0.63 fJ measured; device capacitance also shrinks at low V). Delay grows about 10x per 100 mV below about 0.7 V.

## Results: minimum-energy point (model, not a measurement)
Model of one operation: a path 20 gates deep, a fraction alpha of gates switch per operation, every gate leaks for one full clock period (period = 20 x FO4 delay). E_op = alpha x 20 x E_switch + 20 x P_leak x 20 x t_delay.

| Corner | alpha | Best supply | Energy per op vs 1.8 V | Slowdown vs 1.8 V |
|---|---|---|---|---|
| tt | 0.02 | 0.60 V | 10.5x less | 506x |
| tt | 0.05 | 0.50 V | 12.3x less | 4,006x |
| tt | 0.10 | 0.50 V | 14.8x less | 4,006x |
| tt | 0.20 | 0.45 V | 17.1x less | 12,200x |
| ff | 0.10 | 0.60 V | 9.1x less | 107x |
| ss | 0.10 | 0.50 V | 14.8x less | 31,566x |

alpha = 0.5 rows (tt, ff) hit the lowest resolvable supply, so their minimum is only an upper bound; not reported as results (full table in `mep.json`).

Reading for the post: on open 130 nm silicon, the energy-optimal supply for logic is around 0.5 V, about 10 to 17 times less energy per operation than at the nominal 1.8 V, and the price is speed: hundreds to tens of thousands of times slower depending on corner. That trade is the transistor-level fact the post builds on.

## Reproducibility checks
- Delay: one 50%-50% delay goes negative at low V (switching point drifts from VDD/2 with slow edges); we report the mean of rise and fall, which cancels it.
- Leakage: the in-transient estimate agreed with DC leakage within about 1% at <= 0.7 V, but was noise-dominated (even negative) at high V where periods are a few ns. We use DC leakage everywhere; at high V it is negligible in the energy.
- Unresolvable points: where leakage over the period is over half the measured energy, switching energy cannot be separated reliably; those points are dropped and listed (tt and ss: 0.25 to 0.35 V; ff: 0.35 to 0.45 V).
- Failed points: 0.2 V on all corners and ff at 0.25 and 0.3 V did not settle within the retry limit. Not used.
- Minimum-energy points for alpha 0.02 to 0.2 lie inside the resolved range on every corner.
- Not yet checked: comparison of the minimum-energy supply with the near-threshold literature (Calhoun et al. 2005); sensitivity to logic depth L (20 assumed); temperature other than 27 C.

---

# M3a steps 3 and 4: full adder, and an 8-bit multiply-add built from it (2026-11-19)

## Full adder (simulated, transistor level)
28-transistor mirror adder, each output loaded by 4 inverters, 16 random input patterns per point (fixed seed), every output checked against A+B+Cin at the end of every period. 3 corners x 13 supplies, 486 s. Code `m3a_fulladder_sweep.py`, data `fulladder_sweep.csv`.

| | tt | ff | ss |
|---|---|---|---|
| Carry delay at 1.8 V | 136 ps | 99 ps | 204 ps |
| Carry delay at 0.5 V | 606 ns | 73 ns | 6.7 us |
| Energy per addition at 1.8 V (random data) | 14.9 fJ | 15.2 fJ | 14.7 fJ |
| Energy per addition at 0.5 V | 0.82 fJ | 0.78 fJ | 0.78 fJ |

- Functional check: 37 of 39 points correct on all 32 output samples. ff at 0.35 V gave 4 wrong outputs and ff at 0.3 V did not complete; both excluded. Read: the fast corner, which leaks most, stops computing correctly first as voltage drops; this adder (unsized, minimum widths) needs at least 0.4 V on ff.
- Switching energy is resolvable down to 0.5 V (tt, ss) and 0.7 V (ff); below that leakage over the long period is over half the measured energy, so those points are not used for energy.

## 8-bit multiply-add (a model built on the simulated adder, not a full simulation)
Composition: 8x8 array multiplier (56 adders) + 16-bit ripple accumulator (16 adders) = 72 adders on random data; critical path about 30 carry delays; all adders leak for the whole cycle. The 64 AND gates, wiring and registers are not counted, so these are lower bounds. Code `m3a_mac_compose.py`, data `mac.json`.

| Corner | Energy per MAC at 1.8 V | Lowest resolvable supply | Energy per MAC there | Saving | Slowdown | MACs per second per unit there |
|---|---|---|---|---|---|---|
| tt | 1.07 pJ | 0.50 V | 80 fJ | 13.4x | 4,454x | 55,000 |
| ss | 1.06 pJ | 0.50 V | 85 fJ | 12.5x | 32,803x | 5,000 |
| ff | 1.11 pJ | 0.70 V | 145 fJ | 7.7x | 31x | 11 million |

The minimum-energy point for the MAC is at or below the lowest resolvable supply on every corner (the curve is flat between 0.6 and 0.5 V on ss, still falling on tt), so the "energy there" column is an upper bound on the true minimum. Consistent with the inverter result (about 0.5 V).

For the post: one 8-bit multiply-add on open 130 nm silicon costs about 1 pJ at the standard 1.8 V and under 0.1 pJ near 0.5 V, at thousands of times lower speed. Qwen3-8B needs about 7.6 billion multiply-adds per token (M2), so on this silicon that is about 8 mJ per token at 1.8 V for the arithmetic alone, before any memory traffic (M3b) and before scaling to modern processes (M4). These are back-of-envelope products for the team, not post numbers yet.

## Reproducibility checks
- Every energy point comes with a functional check; wrong or incomplete points are excluded and named.
- Unresolvable low-voltage points excluded and the bound stated, not extrapolated.
- Lower bound stated: AND gates, wires, registers and clocking are missing from the MAC.
- To do in M4: compare with published per-operation energy tables (Horowitz, ISSCC 2014) after scaling; that source has not been opened yet, so no comparison is claimed here.
