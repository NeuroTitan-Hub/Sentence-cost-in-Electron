# M3b: transistors, memory, 2026-11-19

Status: on-chip SRAM simulated and published off-chip figures collected (`sources.md`). The full M4 chain still has to use them.

## Setup
- ngspice 47, sky130 (volare c6d73a35…), 27 C, corners tt / ff / ss, 10 supplies from 1.8 V to 0.5 V.
- **The cells use SkyWater's own SRAM transistors**: `special_nfet_pass` 0.14/0.15, `special_nfet_latch` 0.21/0.15, `special_pfet_latch` 0.14/0.15 um. These are the only sizes the PDK allows for them.
- **One column**: 128 six-transistor cells on a bit-line pair, with precharge and equaliser, a word-line driver on its own supply, and a write driver.
- Junction geometry is the PDK xschem default. That probably overstates junction capacitance a little, because real SRAM layouts share contacts. Bit-line wire capacitance is not included.
- **Read** takes two passes. First we measure the time for the bit line to fall by 100 mV, a typical sense-amplifier input; then we open the word line for only that long. Energy is everything drawn from the precharge and word-line supplies over the cycle, including restoring the bit line.
- **Write** pulls one bit line to 0 and checks that the cell flips.
- **Checks on every point**:
  - a read must not flip the cell;
  - a write must flip it;
  - the bit lines must be restored by the end of the cycle;
  - for hold leakage (DC), the cells must sit in their stored states (checked on two cells).
- Code: `m3b_sram_sweep.py`. Data: `sram_sweep.csv`. 30 points in 1,947 s, plus 2 retries.

## Results (energy per bit, per column access)

| | tt | ff | ss |
|---|---|---|---|
| Read energy at 1.8 V | 14.8 fJ | 14.6 fJ | 14.9 fJ |
| Read energy at 0.9 V | 4.3 fJ | 4.4 fJ | 4.1 fJ |
| Read energy at 0.5 V | 1.6 fJ | 2.0 fJ * | fails |
| Write energy at 1.8 V | 77 fJ | 78 fJ | 74 fJ |
| Read access (word line to 100 mV) at 1.8 V | 65 ps | 50 ps | 93 ps |
| Read access at 0.5 V | 274 ns | 15 ns | fails |
| Column leakage (128 cells) at 1.8 V | 2.8 nW | 32 nW | 2.1 nW |

\* At ff 0.5 V leakage over the cycle is 63% of the measured energy, so the switching part is **not resolvable** there. That point is excluded from any minimum-energy claim.

**Failures (real circuit behaviour, reported as is)**: on the slow corner the **write fails at 0.8 V and below**. The weak pass gate can no longer overpower the cell. This write-margin limit is the usual reason SRAM cannot follow logic down to low voltage. The typical and fast corners still write at 0.5 V in this simulation, but we did no mismatch (Monte Carlo) analysis, and mismatch is what sets SRAM limits on real chips. So "works at 0.5 V" is not claimed.

**Fixed along the way (not physics)**:
- Two points first failed because our cycle left the bit lines too little time to recharge. They passed with a 4× longer cycle: tt 0.9 V and ss 1.0 V.
- Lumping the 127 idle cells into one instance with a multiplier did not reproduce the bit-line load (3.2 vs 6.8 fC), so all 128 cells are explicit.
- The bit-line swing came out at 110-160 mV instead of exactly 100 mV, because the word line takes a little time to turn off. The energies include that overshoot.

## What it means

- Writing a bit costs about 5× more than reading one, because a write swings the bit line fully and a read swings it only about 0.1 V.
- **Reading one 8-bit weight from this column costs about 0.12 pJ at 1.8 V**: 8 × 14.8 fJ. That is about a tenth of our 8-bit multiply-add (1.25 pJ, M3a corrected). So once data sits right next to the arithmetic in SRAM, arithmetic dominates.
- **Off-chip is a different world.** GDDR6 at 7.5 pJ per bit (Micron, vendor claim) is about **500×** our SRAM column's read energy per bit, and HBM2 at 3.9-4.0 pJ is about **270×**.
  - For Qwen3-8B, the 16 GB of weights cannot sit in on-chip SRAM, so every decode step reads them from DRAM.
  - That is why memory traffic, not arithmetic, sets the energy per token in M4.

## Reproducibility check against a published number

Horowitz (ISSCC 2014, Figure 1.1.9) gives 10 pJ for a 64-bit read from an 8 KB cache at 45 nm, 0.9 V. That is 156 fJ per bit. Our column at 0.9 V is 4.3 fJ per bit, about 36× lower. That is the expected direction: our number is the bit cell and bit line only. It leaves out:
- the address decoder;
- sense amplifiers;
- column multiplexing;
- the wires that carry data to and from the array (often the largest part);
- bit-line wire capacitance.

So **our SRAM figure is a floor for the cell and bit line, not the cost of a real cache read**. M4 will carry both, with Horowitz's figure as the "real cache" end of the range.

## Still open
- Sense amplifier and decoder energy (simulate one, or carry Horowitz's figure as the range).
- Monte Carlo on cell mismatch for the low-voltage claim.
- Opened sources for HBM3, HBM3e and GDDR7 energy per bit.
