# M2: tokens to arithmetic, 2026-11-19

Status: done for the Qwen3 family; Reproducibility check passed. Code `m2_ops.py`, output `m2_ops.json`. Runs in 13 s.

## What it counts
Per token, from each model's `config.json` (Qwen3-0.6B, 8B, 14B, 32B; Qwen3-8B is Deng et al.'s main model):
- multiply-adds (MACs): projections and MLP (fixed per token), attention (grows with context length), output head;
- memory traffic: weights (read once per decoding step, shared across a batch) and the KV cache (read in full for every token, per sequence).
bf16 storage as in the configs. Elementwise work (norms, softmax, activations, rotary) is under 1% and listed separately.

## Results
| Model | MACs per token, short context | at 8,192 tokens of context | attention share at 8,192 | KV-cache share of memory traffic at 8,192 (batch 1 / batch 64) |
|---|---|---|---|---|
| Qwen3-0.6B | 0.60 G | 1.54 G | 61% | 44% / 98% |
| Qwen3-8B | 7.57 G | 9.98 G | 24% | 7% / 84% |
| Qwen3-14B | 13.99 G | 17.35 G | 19% | 5% / 75% |
| Qwen3-32B | 31.98 G | 40.57 G | 21% | 3% / 68% |

Reading one FLORES sentence with Qwen3-8B (prefill, M1 token counts): English 204.5 GMAC per sentence. Odia 9.86x, Shan 10.6x, Burmese 9.1x. For sentences this short the arithmetic tracks the token count almost one to one (Odia: 9.70x tokens, 9.86x MACs).

Readings for M4 (not post copy yet):
- Arithmetic per token barely depends on language; token count does. So for reading text, the language gap in arithmetic is essentially the token gap.
- Memory traffic is where context length bites. When a server batches many requests (as Deng et al.'s vLLM setup does, batch 16 to 512), weights are shared but every sequence reads its own KV cache, so at long context the KV cache becomes most of the memory traffic (84% for Qwen3-8B at 8,192 tokens, batch 64). Languages that need more tokens fill the cache faster, so each later token costs more to move. That is a concrete, countable mechanism behind Deng et al.'s per-token energy gap; M4 tests whether it is large enough.

## Reproducibility check
- FLOP counter: our own minimal PyTorch model in the Qwen3 layout (random weights, nothing downloaded), one decoding step at context 1, 64 and 300: counted MACs equal the formula exactly (0 difference).
  Limit: this checks the arithmetic of the formula against an implementation of the same layout; it does not prove Qwen's own code has no extra operations. Mitigation below.
- Parameter totals match Qwen's published sizes: Qwen3-8B 7.568 B counted + 0.622 B input embedding = 8.19 B (model card: 8.2B); Qwen3-0.6B (tied embeddings) 0.596 B (model card: 0.6B). Non-embedding: 6.946 B vs card 6.95B (8B), 0.440 B vs card 0.44B (0.6B). Checked against the cards 2026-11-19.
- Not modelled yet: batching effects on compute efficiency, quantised weights (int8/int4 halve or quarter the weight bytes), speculative decoding. Noted for M4.
