# M5: red team, first pass, 2026-11-19

Method: list every claim the post might make, attack each one, and record what survives. Verdicts:
- **Holds**: survived the attacks we ran.
- **Weakened**: true only with the stated narrowing.
- **Do not claim**: not supported.

Each verdict also lists what we have *not* tried. A claim that holds is not proven, only not yet broken.

## Ledger

| # | Claim | Evidence | Attack | Verdict |
|---|---|---|---|---|
| 1 | The same sentence takes very different token counts across languages (Qwen3: up to 10.4× English, Shan) | M1, 204 languages, 8 tokenizers, bootstrap CIs, 8/8 spot counts | Known since Petrov 2023, Deng 2026 | **Holds, but not new.** Cite, don't present as a finding |
| 2 | The gap is a property of the tokenizer, not the language | M1: Odia 1.37× (BLOOM) to 13.3× (GPT-2); UTF-8 bytes give 2.7× | A tokenizer comes with its model's training; you can't swap it freely | **Holds, narrowed**: "same text, different tokenizers, very different counts". The chain's "Qwen3-8B with BLOOM's tokenizer" numbers are hypothetical and must be labelled that way |
| 3 | M2's MAC formula is exact | Equals PyTorch's FLOP counter at 3 contexts; parameter count matches the model card | Our own minimal implementation, not vLLM kernels (which may recompute or fuse) | **Holds** for the model's arithmetic, not for what a given kernel executes |
| 4 | An 8-bit multiply-add on sky130 costs about 1.25 pJ at 1.8 V and about 0.1 pJ near 0.5 V | Full adder simulated transistor by transistor; 72-adder composition | (a) Junction capacitance was missing: found and fixed. (b) AND gates, wires and registers excluded. (c) No temperature sweep. (d) Mismatch Monte Carlo 2026-11-21, 50 copies x 3 corners x 6 supplies: no change at >= 1.0 V; at 0.5 V typical, clocking for the slow copies raises it from 101 to 134 fJ (+32%) | **Weakened**: a *lower bound* at 27 C. Say "at least"; at 0.5 V quote 0.13 pJ with variation |
| 5 | Lowest-energy supply is about 0.5 V on sky130 | Inverter MEP model and adder compose; same with and without junctions | At the edge of what we can resolve (leakage swamps the measurement below it); depends on assumed logic depth and activity. Monte Carlo (2026-11-21): with a 3 sigma clock the minimum moves to 0.6 V (typical 123 fJ, slow 119 fJ) and 0.7 V on the fast corner (182 fJ) | **Weakened further**: "about 0.5 V for one ideal adder; 0.6-0.7 V once chip-to-chip variation is counted". Best saving over 1.8 V about 10x, not 12x |
| 6 | Reading a bit from on-chip SRAM costs about 15 fJ at 1.8 V | SkyWater SRAM devices, 128-cell column, read/write/restore checks | Excludes decoder, sense amp, wires; Horowitz's real 8 KB cache is about 36× higher per bit | **Weakened**: "the cell and bit line alone". Never compare it with off-chip DRAM as if it were a whole cache |
| 7 | Off-chip memory costs hundreds of times more per bit than on-chip | GDDR6 7.5 pJ (vendor), HBM2 3.9-4.0 pJ (model) vs our 15 fJ | Ours is a floor (see 6). Against Horowitz's whole cache (156 fJ per bit at 45 nm), DRAM at 20-41 pJ is still 130-260× | **Holds** as "100× or more" |
| 8 | Counted KV traffic plus the memory-limited batch reproduces Deng et al.'s per-language ordering (rank correlation about 0.9) | M4 test 1, no fitted parameter, full sweep reported; English prompt length now measured (220 tokens), giving 0.937 | **Mean context length alone ranks just as well (0.937 at the measured prompt).** Their per-token values have 2 decimals (24 levels, about ±5% per ratio) | **Weakened**: "consistent with" their explanation, not "proves" it. The ranking does not single out our mechanism |
| 9 | A large part of the measured per-token gap is time (power burned while a small batch waits), not energy per bit | The bits-only floor tops out at 2.9-4.6× vs 8.1× measured (3.1× at the measured prompt, GDDR6); the time model tops out at 8.3-12.7× (10.7×) | Two bracketing models, neither fitted. The truth sits between them, and we have not measured how it splits | **Weakened**: "counted bit and arithmetic energy alone does not reach the measured gap; time-dependent power must make up the rest". No percentage |
| 10 | Alone, with short texts, energy follows token count one to one | M4 chain: weights are 94-98% of energy | Holds for all four Qwen3 sizes. Breaks for long documents (weights fall to 62%) | **Holds**, with "short texts" kept in |
| 11 | On a busy server the energy gap grows larger than the token gap (Odia 9.7× tokens, 16-24× energy) | M4 chain; grows roughly with the square of the token ratio | Size-dependent: Qwen3-0.6B gives 61×, 32B gives 20× (Odia, long document). Counts KV at off-chip cost even if cached. Floor, not bill | **Holds**, with the model size named and "lower bound on arithmetic plus memory energy, not a GPU's bill" |
| 12 | No language in our set is cheaper than English | All 204, all scenarios | Chinese is 1.00×; within rounding it ties | **Holds** as "none clearly cheaper"; Chinese ties |
| 13 | Anything about today's chips (4-5 nm, HBM3) | none | No opened source | **Do not claim** |
| 14 | Deng et al.'s middle-cost languages are overpredicted because of their steady-state window | Their appendix: the window needs more than 2 × batch requests and a full batch | Not testable without their per-request data | **Do not claim**; list as one of three open explanations |
| 15 | Minimum-size sky130 SRAM cells cannot follow logic down to 0.5 V | Monte Carlo, 2026-11-21: 1,000 cells per corner (tt, ff, ss, sf, fs with mismatch), 9 supplies, hold and read margin (Seevinck), static write trip, timed write. `m5/sram_mc_summary.csv` | Read failures (margin <= 0) at 0.5 V: tt 69.6%, ff 33.5%, ss 100%, sf 100%; fs write failures 13-16% at 0.5-0.7 V. 6-sigma margin (99% yield of 1 MB, Gaussian tail assumed, no redundancy or ECC) needs read 1.0 V tt, 1.2 V ss, 1.8 V sf; write 1.4 V fs. Not attacked: Gaussian extrapolation beyond 1-in-1,000; ideal bit line (no driver, no column load); 27 C only. Open: M3b column test failed ss write at <= 0.8 V, but the cell alone writes at 0.7 V with an ideal bit line, so M3b's limit is probably the write path; not confirmed | **Holds**: "on this process, minimum-size SRAM needs about 1 V on the typical corner and 1.4-1.8 V on every corner; the 0.5 V saving applies to logic, not memory" |

## Fixes made during this pass
- **Spearman was not tie-aware.** Fixed; the numbers barely moved (0.942 → 0.943 at baseline). With 90 ties among 117 measured values it had to be checked.
- **A wrongly described check.** "122/122 rows re-total" was not independent, because their totals are derived from rounded per-token values. M4 README corrected.
- **Ranking credited only to our model.** Simple baselines added to `m4_deng_test.py`; results are in `deng_test.json` under `baselines_spearman`.
- **English prompt length measured**: 212 tokens raw, 220 with the chat wrapper, rebuilt from their Appendix B template (`m4/m4_belebele_prompt.py`). Test 1 rerun with 220 as the baseline: rank correlation 0.937, median error 1.44×, top prediction 10.7× vs 8.1× measured. Nothing in the ledger changes verdict.
- **Stale input caught on the rerun**: test 1 still used the pre-junction multiply-add energy (1.07 pJ). Fixed to 1.25 pJ; the floor variant's top ratio moved from 3.2-5.1× to 2.9-4.6×.
- **Prompt construction confirmed** from their Appendix B: instruction, passage, question and options are all in the target language, and the passages come from FLORES. So scaling by our FLORES token ratio is the right form. The English prompt length is now measured (220); the 50-500 sweep stays in the JSON.

## Not attacked yet (needed before publishing)
- Monte Carlo mismatch: SRAM done 2026-11-21 (claim 15). Logic done 2026-11-21 21:10 (claims 4 and 5): 0 simulation errors in 900 copies; 1 wrong sum (fast corner, 0.45 V). Only 50 copies per point, so the 3 sigma clock rests on a rough sigma.
- Temperature: everything is at 27 C; chips run hotter, and leakage rises fast with temperature.
- Prompt length in other languages: we measured English only and scale by the FLORES ratio. The machine-translated instruction and primer (about a quarter of the English prompt) may not scale that way. Measuring it would need the other 116 Belebele files and Deng et al.'s translated instructions, which are not released yet.
- A second reader of every number in the post: `m6/popper_check.py` now recomputes them all from these files.
- External reviewers: none before publishing; corrections will be dated in the post.
