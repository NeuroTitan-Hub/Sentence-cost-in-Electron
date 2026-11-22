# M4: assembly and the test against Deng et al.

Status 2026-11-19: test 1 done, now with the English prompt length measured; physics-floor chain done for all 204 languages. Scaling to modern chips still open (no opened source yet).

## Test 1: does counted traffic predict their measured per-language energy?

**Their data.** Deng et al., arXiv 2606.21869v3, Table 5:
- Qwen3-8B on Belebele, 122 languages, one NVIDIA L40S, vLLM, maximum batch 256, steady-state GPU energy.
- We extracted the table from the arXiv HTML into `deng2026_table5.csv`.
- Their total-energy column is their rounded J/token (2 decimals) × output tokens, exactly so for 114 of 122 rows. So it is **not** an independent check, and their per-token values have only 24 distinct levels. English's 0.10 J/token carries about ±5% rounding, and so does every ratio to English.
- English works out to 167,621 / 900 questions = 186 tokens per response, matching their Table 1.

**What they say causes the gap**, in words: long outputs fill the KV cache, the server runs smaller batches, and each token costs more.

**What we did**: turned that sentence into counted numbers with `m4_deng_test.py`. Nothing is fitted to their energy measurements. Inputs:
- From their table: output tokens per response.
- Prompt length: the English prompt × our FLORES Qwen3 token ratio (M1). The English prompt is now **measured**, not assumed (next section).
- From our M2: MACs and bytes per token from the model config.
- From NVIDIA's page (opened): L40S memory 48 GB, bandwidth 864 GB/s, BF16 362 TFLOPS dense, power 350 W.
- Batch = min(256, what fits in the KV budget).
- Step time = roofline max(bytes / bandwidth, FLOPs / peak).

**The English prompt, measured** (`m4_belebele_prompt.py`, output `belebele_prompt.json`):
- We rebuilt their prompt from Appendix B: the instruction (ending with the line "#### N"), then the labelled passage, question and four numbered options, then "Let's think step by step." Text taken from Belebele's English file (facebook/belebele, revision 7899cdfa, CC BY-SA 4.0; 836 KB).
- Qwen3 tokenizer, all 900 questions: **mean 212 tokens, 220 with Qwen's chat wrapper** (5-95%: 167-284). The passage alone is 100 tokens; the instruction and primer add 58.
- Not stated in the paper, so these are our choices: whether vLLM applied the chat template (we take 220, the larger), and the line breaks between parts (a few tokens either way).
- The 900 questions use only 488 distinct passages. This matters for open explanation 2 below.
- One limit stays: we measured English only. Other languages are still English × our FLORES ratio. Their passages, questions and options are Belebele translations and should scale about like FLORES, but the instruction and primer (about a quarter of the English prompt) were machine-translated by Deng et al. and may not.

**Result**: 117 languages overlap (5 romanised variants are not in FLORES-200). The headline row is the measured prompt: 220 English prompt tokens, 23 GB of KV budget.

| English prompt tokens (assumed) | Rank correlation (Spearman) | Median error per language | Highest predicted | Highest measured |
|---|---|---|---|---|
| 50 | 0.89 | 1.19× | 8.7× | 8.1× (Tibetan) |
| 150 | 0.92 | 1.30× | 10.0× | 8.1× |
| **220 (measured)** | **0.94** | **1.44×** | **10.7×** | **8.1×** |
| 300 | 0.94 | 1.58× | 11.4× | 8.1× |
| 500 | 0.95 | 1.85× | 12.6× | 8.1× |

The full sweep, including KV budgets of 20, 23 and 26 GB, is in `deng_test.json`.

Baseline (measured 220 prompt tokens, 23 GB):

| Language | Output tokens | Predicted | Measured |
|---|---|---|---|
| Tibetan | 3,937 | 8.3× | 8.1× |
| Southern Pashto | 4,164 | 6.3× | 8.0× |
| Tigrinya | 3,695 | 6.4× | 6.9× |
| Shan | 4,667 | 10.7× | 6.7× |
| Amharic | 2,276 | 4.7× | 4.6× |

**What this supports, and what it does not (M5 red team)**:
- Counting KV-cache traffic, plus the batch limit that GPU memory imposes, reproduces the *ordering* of 117 languages (rank correlation about 0.9) and the size of the top gap, with no fitted parameter.
- **But simpler predictors rank almost as well**, so the ranking does not single out our mechanism. Rank correlation with the measured ratio (tie-aware):
  - output length alone: 0.84;
  - prompt length alone: 0.93;
  - mean context length alone: 0.94, the same as our full model (0.937 for both at the measured prompt length).
- In this regime our model is close to "cost per token is proportional to context length". What it adds is the *reason*: KV reads dominate once memory limits the batch. That reason comes from counted bytes and the memory size, not from a better fit.
- The claim we can make: Deng et al.'s explanation is quantitatively *consistent* with counted traffic. The claim we cannot make: that we proved it.

**Where it fails (reported, not tuned away)**:
- **Middle-cost languages are overpredicted by 2.2-2.5×** (at the measured prompt length): Burmese, Punjabi, Odia, Gujarati, Maltese and Lao. Their outputs are 1,300-2,400 tokens, yet measured cost is only 1.4-3× English. The model says the KV cache should already be squeezing the batch for them; the measurements say it is squeezing less.
- **Candidate explanations, not yet tested**:
  1. Deng et al. measure only in windows where the batch is full at 256, which may select shorter-context periods for these languages.
  2. Prefix caching: the 900 questions share 488 passages (measured), so if vLLM's prefix cache was on, part of many prompts was never recomputed. Prefill is a small part of decode-heavy runs, so we expect this to be minor; not tested.
  3. The real token-weighted context differs from our mean-based one.
- **We cannot separate these without their per-request data.** They say they will release it; no link yet.
- **Absolute level**: matching English would need 590-1,130 W across the sweep (861 W at the measured prompt) at our ideal roofline speed, but the L40S is capped at 350 W. So the real step time is about 1.7-3.2× slower than ideal (2.5× at the measured prompt), i.e. realised throughput roughly 30-60% of peak. This affects absolute joules, not the ratios above.
- **The "physics floor" variant does not reproduce the gap.** It counts only the energy of moving bits and doing MACs (GDDR 7.5-14 pJ/bit, our MAC energy), with no idle or static power. That variant predicts at most 2.9-4.6× where 8.1× was measured (with the corrected 1.25 pJ multiply-add; it was 3.2-5.1× with the pre-correction 1.07 pJ, which the script still used until this rerun). So a large part of the measured per-token gap is **time**: the GPU burns power while a small batch waits on memory, not because each bit costs more. This may be the most useful single finding for the post, and it needs a red-team pass (M5).

## The chain: the same content in 204 languages (`m4_chain.py`)

**Task, the same meaning in every language**: read a text of N FLORES sentences, then write a 5-sentence answer. Inputs:
- token counts per language from M1 (Qwen3 tokenizer, with M1's 95% intervals);
- MACs and bytes from M2's formula (Qwen3-8B, bf16);
- energy per multiply-add and per off-chip bit from M3a/M3b and the opened sources.

This is the **physics floor**: arithmetic plus moving bits, with no idle power (test 1 shows that part is large on a real GPU).

**Configurations**:
- **Reference**: 45 nm fp16 multiply-add, 1.5 pJ (Horowitz 2014), plus GDDR6 at 7.5 pJ per bit (Micron).
- **Range**: every combination of three multiply-add energies and three memory energies, plus M1's token-count interval.
  - Multiply-add: our sky130 int8 at 1.25 pJ, Horowitz int8 at 0.23 pJ, Horowitz fp16 at 1.5 pJ.
  - Memory: HBM2 3.95, GDDR6 7.5 and DRAM 30 pJ per bit.

**Reproducibility checks**:
- The closed-form sums equal a step-by-step loop in four test cases, with zero difference.
- Hand check: one English answer, alone (batch 1), reads 15.1 GB of weights 138 times. That is 138 × 15.1 GB × 8 × 7.5 pJ = 125 J, matching the script's weight part.
- On the GPU used by Deng et al., the same answer takes about 2.4 s at bandwidth limit. At up to 350 W, that is several hundred joules, so the floor is below reality, as it should be.

**Energy ratio to English** (reference configuration; range in brackets):

| | Tokens | 1 sentence, alone | Page (20 sentences), busy server (batch 64) | Long document (200), busy server |
|---|---|---|---|---|
| French | 1.58× | 1.58× | 1.66× [1.61-1.84] | 1.74× [1.69-2.19] |
| Chinese (Simplified) | 1.00× | 1.01× | 1.00× | 1.00× |
| Southern Pashto | 2.80× | 2.80× | 3.20× [3.03-4.13] | 3.67× [3.45-6.02] |
| Hindi | 4.42× | 4.42× | 5.63× [5.14-8.34] | 7.01× [6.40-14.0] |
| Bengali | 5.03× | 5.04× | 6.65× [6.02-10.3] | 8.51× [7.71-17.8] |
| Odia | 9.70× | 9.75× | 16.5× [13.9-31.5] | 24.2× [21.0-62.9] |
| Shan | 10.43× | 10.5× | 18.3× [15.3-36.0] | 27.4× [23.5-72.9] |

Across all 204 languages:

| | Median | 5× or more | 10× or more | Most expensive |
|---|---|---|---|---|
| 1 sentence, alone | 2.1× | 22 languages | 1 language | Shan, 10.5× |
| Page, busy server | 2.3× | 35 languages | 11 languages | Shan, 18.3× |
| Long document, busy server | 2.5× | 36 languages | 16 languages | Shan, 27.4× |

No language is cheaper than English.

**What the chain shows (the "why" the post is about)**:

1. **Alone, with short texts, energy follows token count one to one.** With one user, every new token reads all 15 GB of weights from memory. For a sentence or a page, that is 94-98% of the energy, so a language that needs 9.7× the tokens costs 9.7× the energy. Even alone, a long document breaks this: weights fall to 62% of English's energy, and Odia rises to 15.4× (range 13.8-16.3).
2. **On a busy server, the gap grows beyond the token count.**
   - Batching shares the weight reads across 64 users, so what is left is what each user cannot share: their own arithmetic and their own KV cache.
   - Attention and KV traffic grow with *context × output*, and both get longer by the token ratio. So that part grows roughly with the *square* of the token ratio.
   - Result: Odia rises from 9.7× (alone) to 16-24× (busy server), and Shan from 10.4× to 18-27×. The gap can be up to 2.6× bigger than the token gap.
3. **This is the floor, not the bill.** A real GPU also burns power while it waits. Test 1 shows that this waiting is what turns these ratios into Deng et al.'s measured 8× per-token gap at batch 256.

**Not in the chain (stated)**:
- idle and static power (see test 1);
- on-chip caches and SRAM. Weights stream from off-chip each step; KV reads are counted at off-chip cost, which overstates KV energy if a cache holds part of it;
- the effect of memory capacity on batch size (test 1 covers it).
- **Modern process nodes.** We have opened no source for the energy per operation or per bit of the 4-5 nm GPUs Deng et al. used. So the chain stops at 45 nm and 2017-2020 memory, and **no claim about today's chips is made**.

## Reproducibility checks done

- Their table re-totalled: consistent, but only because the totals are derived from the rounded per-token values (see above). Not an independent check.
- L40S specification read from NVIDIA's own page, not an aggregator. One aggregator listed BF16 at half the NVIDIA figure, and we did not use it.
- Swept every unknown we know of (prompt length, KV budget, memory energy per bit) and reported all runs. The prompt length is now also measured; the sweep stays in the JSON.
- Caught on the rerun: the test 1 script still used the M3a multiply-add energy from before the junction fix (1.07 pJ). Now 1.25 pJ. It only affects the floor variant.

## Still open in M4

- Scaling to modern nodes: needs opened sources for 4-5 nm energy per operation and HBM3/GDDR6X/GDDR7 per bit.
- Add idle power to the chain (from test 1's implied efficiency) so the interactive calculator can show floor vs realistic.
- Opened source for HBM3 or GDDR7 energy per bit (none yet).
- Ask Deng et al. for their per-request data once released.
